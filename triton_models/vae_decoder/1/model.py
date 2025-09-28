#!/usr/bin/env python3
"""
VAE Decoder for Triton Inference Server

FLUX pipeline의 VAE decoder 로직을 구현합니다.
latent space에서 RGB 이미지로 변환하는 GPU 모델입니다.

Based on flux_pipeline.py lines 1004-1007, 529-542
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

# ===================================================================
# 공통 함수들 (common_functions_template.py에서 복사)
# ===================================================================

def check_triton_availability():
    """Triton Python Backend 가용성 체크 및 Mock 설정"""
    try:
        import triton_python_backend_utils as pb_utils
        return True, pb_utils
    except ImportError:
        import logging
        logging.warning("Triton backend not available - running in test mode")

        class MockPbUtils:
            class Tensor:
                def __init__(self, name, data): pass
            class InferenceRequest: pass
            class InferenceResponse:
                def __init__(self, output_tensors=None, error=None): pass
            class TritonError:
                def __init__(self, message): pass

        return False, MockPbUtils()

def initialize_model_base(args, model_name: str, required_modules):
    """모든 모델의 공통 초기화 로직"""
    import json
    import logging
    logging.basicConfig(level=logging.INFO, format=f'[{model_name}] %(message)s')

    # Triton 가용성 체크
    triton_available, pb_utils = check_triton_availability()

    # 모델 config 파싱
    model_config = json.loads(args['model_config'])
    params = {}
    for param in model_config.get('parameters', []):
        params[param['key']] = param['value']['string_value']

    return {
        'triton_available': triton_available,
        'pb_utils': pb_utils,
        'model_config': model_config,
        'params': params
    }

def handle_model_error(pb_utils, triton_available: bool, error: Exception, context: str = ""):
    """모델 에러 처리 및 응답 생성"""
    import logging
    error_msg = f"{context} 처리 중 오류: {str(error)}"
    logging.error(error_msg)

    if triton_available:
        return pb_utils.InferenceResponse(
            output_tensors=[],
            error=pb_utils.TritonError(error_msg)
        )
    return None

# ===================================================================

# 필수 의존성
try:
    import torch
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
except ImportError as e:
    print(f"❌ 필수 의존성 누락: {e}")
    sys.exit(1)


class TritonPythonModel:
    """VAE Decoder Triton Model"""

    def initialize(self, args: Dict) -> None:
        """모델 초기화"""
        # 공통 초기화 로직 사용
        init_result = initialize_model_base(args, "VAE_Decoder", ["torch", "diffusers"])

        # 초기화 결과를 인스턴스 변수에 저장
        self.triton_available = init_result['triton_available']
        self.pb_utils = init_result['pb_utils']
        self.model_config = init_result['model_config']
        params = init_result['params']

        # VAE 특화 파라미터 추출
        self.model_name = params.get('model_name', 'black-forest-labs/FLUX.1-schnell')
        self.vae_scale_factor = float(params.get('vae_scale_factor', '8'))
        self.scaling_factor = float(params.get('scaling_factor', '0.3611'))
        self.shift_factor = float(params.get('shift_factor', '0.1159'))

        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📍 Device: {self.device}")

        # VAE 모델 로드
        try:
            self.vae = AutoencoderKL.from_pretrained(
                self.model_name,
                subfolder="vae",
                torch_dtype=torch.bfloat16  # 메모리 최적화
            ).to(self.device)

            # VAE config에서 스케일링 팩터 업데이트 (가능한 경우)
            if hasattr(self.vae.config, 'scaling_factor'):
                self.scaling_factor = self.vae.config.scaling_factor
            if hasattr(self.vae.config, 'shift_factor'):
                self.shift_factor = self.vae.config.shift_factor

            print(f"✅ VAE 로드 완료: scaling_factor={self.scaling_factor}, shift_factor={self.shift_factor}")

        except Exception as e:
            print(f"❌ VAE 로딩 실패: {e}")
            raise

        # 이미지 후처리기 초기화
        self.image_processor = VaeImageProcessor()

        # VAE를 evaluation 모드로 설정
        self.vae.eval()

        print("✅ VAE Decoder 초기화 완료")


    def _unpack_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Latent unpacking 구현 (flux_pipeline.py lines 529-542)

        Args:
            latents: 패킹된 latent 텐서 (batch_size, num_patches, 64)
            height: 목표 이미지 높이
            width: 목표 이미지 너비

        Returns:
            언패킹된 latent 텐서 (batch_size, 16, latent_height, latent_width)
        """
        batch_size, num_patches, channels = latents.shape

        # VAE는 8x 압축을 적용하지만 패킹으로 인해 latent 높이/너비가 2로 나누어떨어져야 함
        height = 2 * (int(height) // (int(self.vae_scale_factor) * 2))
        width = 2 * (int(width) // (int(self.vae_scale_factor) * 2))

        # 패킹 해제: (batch_size, num_patches, 64) -> (batch_size, 16, height, width)
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def _tensor_from_dlpack_or_numpy(self, pb_tensor) -> torch.Tensor:
        """DLPack 또는 numpy를 통한 텐서 변환 (우아한 성능 저하)"""
        try:
            import torch.utils.dlpack
            dlpack_available = True
        except ImportError:
            dlpack_available = False

        if dlpack_available and hasattr(pb_tensor, 'to_dlpack'):
            try:
                return torch.utils.dlpack.from_dlpack(pb_tensor.to_dlpack())
            except Exception:
                pass  # fallback to numpy

        # numpy fallback
        numpy_array = pb_tensor.as_numpy()
        return torch.from_numpy(numpy_array)

    def _tensor_to_pb_tensor(self, tensor: torch.Tensor, name: str):
        """torch.Tensor를 pb_utils.Tensor로 변환"""
        try:
            import torch.utils.dlpack
            dlpack_available = True
        except ImportError:
            dlpack_available = False

        if dlpack_available and tensor.is_cuda:
            try:
                dlpack_tensor = torch.utils.dlpack.to_dlpack(tensor)
                return self.pb_utils.Tensor.from_dlpack(name, dlpack_tensor)
            except Exception:
                pass  # fallback to numpy

        # numpy fallback
        numpy_array = tensor.cpu().numpy()
        return self.pb_utils.Tensor(name, numpy_array)

    def execute(self, requests: List) -> List:
        """배치 추론 실행"""
        responses = []

        for request in requests:
            try:
                # 입력 텐서 추출
                latents_pb = self.pb_utils.get_input_tensor_by_name(request, "latents")
                height_pb = self.pb_utils.get_input_tensor_by_name(request, "height")
                width_pb = self.pb_utils.get_input_tensor_by_name(request, "width")

                # 텐서 변환
                latents = self._tensor_from_dlpack_or_numpy(latents_pb).to(
                    device=self.device, dtype=torch.bfloat16
                )
                height = int(height_pb.as_numpy()[0])
                width = int(width_pb.as_numpy()[0])

                # VAE 디코딩 수행
                with torch.no_grad():
                    # 1. Latent unpacking (flux_pipeline.py lines 529-542)
                    unpacked_latents = self._unpack_latents(latents, height, width)

                    # 2. Scaling 적용 (flux_pipeline.py line 1005)
                    scaled_latents = (unpacked_latents / self.scaling_factor) + self.shift_factor

                    # 3. VAE decode (flux_pipeline.py line 1006)
                    decoded_images = self.vae.decode(scaled_latents, return_dict=False)[0]

                    # 4. 후처리 (flux_pipeline.py line 1007)
                    # VaeImageProcessor.postprocess를 간소화한 버전
                    # RGB 값을 [0, 1] 범위로 클램핑
                    images = torch.clamp((decoded_images + 1.0) / 2.0, 0.0, 1.0)

                # 출력 텐서 변환
                output_tensor = self._tensor_to_pb_tensor(images, "images")
                inference_response = self.pb_utils.InferenceResponse([output_tensor])
                responses.append(inference_response)

            except Exception as e:
                error_response = handle_model_error(
                    self.pb_utils,
                    self.triton_available,
                    e,
                    "VAE Decoder"
                )
                responses.append(error_response)

        return responses

    def finalize(self) -> None:
        """모델 정리"""
        print("🔄 VAE Decoder 정리 중...")
        if hasattr(self, 'vae'):
            del self.vae
        torch.cuda.empty_cache()
        print("✅ VAE Decoder 정리 완료")


# 직접 실행 가능한 테스트 함수 (헌장 요구사항)
def test_vae_decoder():
    """VAE decoder 테스트 함수"""
    print("🧪 VAE Decoder 테스트 시작")

    # DLPack 사용 여부 결정 (테스트 시 분기처리)
    try:
        import torch.utils.dlpack
        dlpack_available = True
    except ImportError:
        dlpack_available = False
    use_dlpack = dlpack_available and torch.cuda.is_available()
    print(f"📊 DLPack 사용: {use_dlpack}")

    # 테스트 모델 설정
    model_config = {
        "parameters": [
            {"key": "model_name", "value": {"string_value": "black-forest-labs/FLUX.1-schnell"}},
            {"key": "vae_scale_factor", "value": {"string_value": "8"}},
            {"key": "scaling_factor", "value": {"string_value": "0.3611"}},
            {"key": "shift_factor", "value": {"string_value": "0.1159"}}
        ]
    }

    # 모델 초기화
    model = TritonPythonModel()
    args = {"model_config": json.dumps(model_config)}

    try:
        model.initialize(args)

        # 테스트 데이터 생성
        batch_size = 1
        height, width = 1024, 1024
        latent_height = height // 8  # VAE scale factor
        latent_width = width // 8
        num_patches = (latent_height // 2) * (latent_width // 2)  # 패킹 고려

        # 패킹된 latent 텐서 생성 (batch_size, num_patches, 64)
        test_latents = torch.randn(batch_size, num_patches, 64, dtype=torch.bfloat16)
        test_height = torch.tensor([height], dtype=torch.int32)
        test_width = torch.tensor([width], dtype=torch.int32)

        print(f"📏 테스트 입력 shape: latents={test_latents.shape}, height={height}, width={width}")

        # Mock request 생성 (triton_available=False일 때)
        triton_available, _ = check_triton_availability()
        if not triton_available:
            class MockTensor:
                def __init__(self, data):
                    self.data = data

                def as_numpy(self):
                    if isinstance(self.data, torch.Tensor):
                        return self.data.numpy()
                    return self.data

            class MockRequest:
                def __init__(self, latents, height, width):
                    self.tensors = {
                        "latents": MockTensor(latents),
                        "height": MockTensor(height),
                        "width": MockTensor(width)
                    }

            # pb_utils.get_input_tensor_by_name 모킹
            import types
            mock_pb_utils = types.ModuleType('pb_utils')

            def mock_get_input_tensor_by_name(request, name):
                return request.tensors[name]

            mock_pb_utils.get_input_tensor_by_name = mock_get_input_tensor_by_name
            sys.modules['triton_python_backend_utils'] = mock_pb_utils

            # 테스트 실행
            mock_request = MockRequest(test_latents, test_height, test_width)

            # 직접 VAE 디코딩 테스트
            with torch.no_grad():
                unpacked = model._unpack_latents(test_latents.to(model.device), height, width)
                print(f"📏 언패킹된 latent shape: {unpacked.shape}")

                scaled = (unpacked / model.scaling_factor) + model.shift_factor
                decoded = model.vae.decode(scaled, return_dict=False)[0]
                images = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)

                print(f"📏 최종 이미지 shape: {images.shape}")
                print(f"📊 이미지 값 범위: [{images.min().item():.3f}, {images.max().item():.3f}]")

        print("✅ VAE Decoder 테스트 완료")

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        raise

    finally:
        model.finalize()


if __name__ == "__main__":
    test_vae_decoder()