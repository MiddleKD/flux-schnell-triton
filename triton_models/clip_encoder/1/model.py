#!/usr/bin/env python3
"""
CLIP Text Encoder for Triton Inference Server

FLUX pipeline의 _get_clip_prompt_embeds 로직을 구현합니다.
텍스트를 CLIP pooled embeddings로 변환하는 GPU 모델입니다.

Based on flux_pipeline.py lines 267-309
"""

import os
import sys
import json
import numpy as np
import logging
from typing import Dict, List, Optional

# ===================================================================
# 공통 함수들 (common_functions_template.py에서 복사)
# ===================================================================

def check_triton_availability():
    """Triton Python Backend 가용성 체크 및 Mock 설정"""
    try:
        import triton_python_backend_utils as pb_utils
        return True, pb_utils
    except ImportError:
        logging.warning("Triton backend not available - running in test mode")

        # Mock pb_utils for testing
        class MockTensor:
            def __init__(self, name, data):
                self.name = name
                self.data = data

        class MockInferenceRequest:
            def __init__(self):
                pass

        class MockInferenceResponse:
            def __init__(self, output_tensors=None, error=None):
                self.output_tensors = output_tensors or []
                self.error_msg = error

        class MockTritonError:
            def __init__(self, message):
                self.message = message

        class MockPbUtils:
            Tensor = MockTensor
            InferenceRequest = MockInferenceRequest
            InferenceResponse = MockInferenceResponse
            TritonError = MockTritonError

            @staticmethod
            def get_input_tensor_by_name(request, name):
                return None

            @staticmethod
            def get_output_tensor_by_name(response, name):
                return None

        return False, MockPbUtils()

def check_dlpack_availability():
    """DLPack 가용성 체크"""
    try:
        import torch.utils.dlpack
        from torch.utils.dlpack import to_dlpack, from_dlpack
        return True, to_dlpack, from_dlpack
    except ImportError:
        logging.warning("DLPack not available - falling back to standard tensor operations")
        return False, None, None

def check_model_dependencies(required_modules: List[str]) -> bool:
    """필수 의존성 체크"""
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        logging.error(f"Missing required modules: {missing_modules}")
        return False
    return True

def extract_model_parameters(model_config: Dict) -> Dict[str, str]:
    """모델 config에서 모든 파라미터 추출"""
    params = {}
    for param in model_config.get('parameters', []):
        params[param['key']] = param['value']['string_value']
    return params

def setup_logging(model_name: str):
    """모델별 로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[{model_name}] %(asctime)s - %(levelname)s - %(message)s'
    )

def create_error_response(pb_utils, error_message: str, triton_available: bool):
    """표준화된 에러 응답 생성"""
    if triton_available:
        return pb_utils.InferenceResponse(
            output_tensors=[],
            error=pb_utils.TritonError(error_message)
        )
    else:
        # Test mode에서는 None 반환
        logging.error(f"Error (test mode): {error_message}")
        return None

def handle_model_error(pb_utils, triton_available: bool, error: Exception, context: str = ""):
    """모델 에러 처리 및 응답 생성"""
    error_msg = f"{context} 처리 중 오류: {str(error)}"
    logging.error(error_msg)
    return create_error_response(pb_utils, error_msg, triton_available)

def setup_device(prefer_cuda: bool = True):
    """최적 디바이스 설정"""
    try:
        import torch
        if prefer_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU device")
        return device
    except ImportError:
        logging.warning("PyTorch not available, device setup skipped")
        return None

def initialize_model_base(args: Dict, model_name: str, required_modules: List[str]):
    """모든 모델의 공통 초기화 로직"""
    # 로깅 설정
    setup_logging(model_name)
    logging.info(f"{model_name} 초기화 시작...")

    # 의존성 체크
    if not check_model_dependencies(required_modules):
        logging.error(f"{model_name} 의존성 체크 실패")
        sys.exit(1)

    # Triton & DLPack 가용성 체크
    triton_available, pb_utils = check_triton_availability()
    dlpack_available, to_dlpack, from_dlpack = check_dlpack_availability()

    # 모델 config 파싱
    model_config = json.loads(args['model_config'])
    params = extract_model_parameters(model_config)

    # 디바이스 설정
    device = setup_device()

    return {
        'triton_available': triton_available,
        'pb_utils': pb_utils,
        'dlpack_available': dlpack_available,
        'to_dlpack': to_dlpack,
        'from_dlpack': from_dlpack,
        'model_config': model_config,
        'params': params,
        'device': device
    }

# ===================================================================

# 필수 의존성
try:
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError as e:
    print(f"❌ 필수 의존성 누락: {e}")
    sys.exit(1)


class TritonPythonModel:
    """CLIP Text Encoder Triton Model"""

    def initialize(self, args: Dict) -> None:
        """모델 초기화"""
        # 공통 초기화 로직 사용
        init_result = initialize_model_base(
            args,
            "CLIP_Encoder",
            ["torch", "transformers"]  # 필수 의존성
        )

        # 초기화 결과를 인스턴스 변수에 저장
        self.triton_available = init_result['triton_available']
        self.pb_utils = init_result['pb_utils']
        self.dlpack_available = init_result['dlpack_available']
        self.to_dlpack = init_result['to_dlpack']
        self.from_dlpack = init_result['from_dlpack']
        self.model_config = init_result['model_config']
        self.device = init_result['device']
        params = init_result['params']

        # CLIP 특화 파라미터 추출
        self.model_name = params.get("model_name", "black-forest-labs/FLUX.1-schnell")
        self.max_length = int(params.get("max_sequence_length", "77"))
        self.embedding_dim = int(params.get("embedding_dim", "768"))

        # CLIP 모델 로드
        logging.info(f"CLIP 모델 로드 중: {self.model_name}")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_name,
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder"
        )
        self.text_encoder.to(self.device)
        self.text_encoder.eval()
        logging.info(f"CLIP 모델 로드 완료: {self.model_name}")

    def execute(self, requests: List) -> List:
        """배치 추론 실행"""
        responses = []

        for request in requests:
            try:
                response = self._process_request(request)
                responses.append(response)
            except Exception as e:
                error_response = handle_model_error(
                    self.pb_utils,
                    self.triton_available,
                    e,
                    "CLIP Encoder"
                )
                responses.append(error_response)

        return responses

    def _process_request(self, request):
        """단일 요청 처리"""
        # 입력 텐서 추출
        input_ids_tensor = pb_utils.get_input_tensor_by_name(request, "input_ids") if TRITON_AVAILABLE else None
        attention_mask_tensor = pb_utils.get_input_tensor_by_name(request, "attention_mask") if TRITON_AVAILABLE else None

        if not TRITON_AVAILABLE:
            # 테스트 모드: 더미 데이터
            batch_size = 1
            input_ids = torch.randint(0, 49407, (batch_size, self.max_length), dtype=torch.long)
            attention_mask = torch.ones((batch_size, self.max_length), dtype=torch.long)
        else:
            # Triton 모드: 실제 데이터
            input_ids = self._tensor_to_torch(input_ids_tensor)
            attention_mask = self._tensor_to_torch(attention_mask_tensor) if attention_mask_tensor else None

        # CLIP 인코딩 수행 (flux_pipeline.py 로직 기반)
        pooled_embeds = self._encode_text(input_ids, attention_mask)

        # 응답 생성
        if TRITON_AVAILABLE:
            output_tensor = self._torch_to_tensor(pooled_embeds, "pooled_embeds")
            return pb_utils.InferenceResponse([output_tensor])
        else:
            return pooled_embeds

    def _tensor_to_torch(self, triton_tensor) -> torch.Tensor:
        """Triton 텐서를 PyTorch 텐서로 변환"""
        if DLPACK_AVAILABLE and hasattr(triton_tensor, 'to_dlpack'):
            # DLPack 사용 (메모리 복사 최소화)
            return torch.utils.dlpack.from_dlpack(triton_tensor.to_dlpack())
        else:
            # Fallback: numpy를 통한 변환
            numpy_array = triton_tensor.as_numpy()
            return torch.from_numpy(numpy_array).to(self.device)

    def _torch_to_tensor(self, torch_tensor: torch.Tensor, name: str):
        """PyTorch 텐서를 Triton 텐서로 변환"""
        if DLPACK_AVAILABLE:
            # DLPack 사용
            return pb_utils.Tensor.from_dlpack(name, torch.utils.dlpack.to_dlpack(torch_tensor))
        else:
            # Fallback: numpy를 통한 변환
            numpy_array = torch_tensor.detach().cpu().numpy()
            return pb_utils.Tensor(name, numpy_array)

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        CLIP 텍스트 인코딩

        flux_pipeline.py의 _get_clip_prompt_embeds 메서드 기반 (lines 267-309)
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            # CLIP 텍스트 인코더 실행
            text_encoder_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )

            # Pooled output 사용 (flux_pipeline.py line 302)
            pooled_embeds = text_encoder_output.pooler_output

            # 데이터 타입 확인
            pooled_embeds = pooled_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

            print(f"📊 CLIP 출력 형태: {pooled_embeds.shape}")  # [batch_size, 768]

            return pooled_embeds

    def finalize(self) -> None:
        """모델 정리"""
        print("🔄 CLIP Encoder 정리 중...")
        if hasattr(self, 'text_encoder'):
            del self.text_encoder
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ CLIP Encoder 정리 완료")


def test_clip_encoder():
    """직접 실행 테스트 (헌장 요구사항)"""
    print("🚀 CLIP Encoder 테스트 모드")
    print("=" * 50)

    # 테스트 환경 설정
    os.environ["TEST_MODE"] = "true"

    # 모델 초기화
    model = TritonPythonModel()
    args = {
        'model_config': json.dumps({
            'parameters': [
                {'key': 'model_name', 'value': {'string_value': 'black-forest-labs/FLUX.1-schnell'}},
                {'key': 'max_sequence_length', 'value': {'string_value': '77'}},
                {'key': 'embedding_dim', 'value': {'string_value': '768'}}
            ]
        })
    }

    try:
        model.initialize(args)

        # 테스트 데이터 생성
        batch_size = 2
        input_ids = torch.randint(0, 49407, (batch_size, 77), dtype=torch.long)
        attention_mask = torch.ones((batch_size, 77), dtype=torch.long)

        print(f"📋 테스트 입력 형태: {input_ids.shape}")

        # 추론 실행
        pooled_embeds = model._encode_text(input_ids, attention_mask)

        # 결과 검증
        expected_shape = (batch_size, 768)
        if pooled_embeds.shape == expected_shape:
            print(f"✅ 출력 형태 검증 통과: {pooled_embeds.shape}")
        else:
            print(f"❌ 출력 형태 오류: {pooled_embeds.shape}, 예상: {expected_shape}")
            return False

        # 값 범위 검증
        min_val, max_val = pooled_embeds.min().item(), pooled_embeds.max().item()
        print(f"📊 출력 값 범위: {min_val:.4f} ~ {max_val:.4f}")

        model.finalize()
        print("🎉 CLIP Encoder 테스트 성공!")
        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    # 직접 실행 시 테스트 모드
    success = test_clip_encoder()
    sys.exit(0 if success else 1)