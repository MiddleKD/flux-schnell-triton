#!/usr/bin/env python3
"""
DIT Transformer for Triton Inference Server

FLUX pipeline의 transformer 호출 로직을 구현합니다.
DIT (Diffusion Transformer)를 사용한 노이즈 예측 GPU 모델입니다.

Based on flux_pipeline.py lines 944-954 (denoising loop의 transformer 호출)
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional

# Triton Python Backend
try:
    import triton_python_backend_utils as pb_utils
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("⚠️ Triton Python Backend not available - running in test mode")

# DLPack 지원 확인 (헌장 요구사항: 우아한 성능 저하)
try:
    import torch.utils.dlpack
    DLPACK_AVAILABLE = True
except ImportError:
    DLPACK_AVAILABLE = False

# 필수 의존성
try:
    import torch
    from diffusers import FluxTransformer2DModel, BitsAndBytesConfig as DiffusersBitsAndBytesConfig
except ImportError as e:
    print(f"❌ 필수 의존성 누락: {e}")
    sys.exit(1)


class TritonPythonModel:
    """DIT Transformer Triton Model"""

    def initialize(self, args: Dict) -> None:
        """모델 초기화"""
        print("🔄 DIT Transformer 초기화 중...")

        # 모델 설정 로드
        self.model_config = json.loads(args['model_config'])

        # 파라미터 추출
        self.model_name = self._get_parameter("model_name", "black-forest-labs/FLUX.1-schnell")
        self.in_channels = int(self._get_parameter("in_channels", "64"))
        self.num_inference_steps = int(self._get_parameter("num_inference_steps", "4"))
        self.use_fp8 = self._get_parameter("use_fp8", "false").lower() == "true"

        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📍 사용 디바이스: {self.device}")

        # DIT Transformer 모델 로드
        print(f"🔄 DIT Transformer 로드 중: {self.model_name} (FP8: {self.use_fp8})")

        # FP8 quantization 설정 (flux_inference.py 방식)
        if self.use_fp8:
            print("🔧 8bit quantization 사용")
            quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
            self.transformer = FluxTransformer2DModel.from_pretrained(
                self.model_name,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=torch.float16
            )
        else:
            print("🔧 FP16 사용")
            self.transformer = FluxTransformer2DModel.from_pretrained(
                self.model_name,
                subfolder="transformer",
                torch_dtype=torch.float16
            )
            self.transformer.to(self.device)
        self.transformer.eval()

        # joint_attention_kwargs 초기화
        self.joint_attention_kwargs = {}

        # guidance_embeds 설정 확인 (flux_pipeline.py 로직 기반)
        self.guidance_embeds = getattr(self.transformer.config, 'guidance_embeds', False)
        print(f"📋 guidance_embeds 설정: {self.guidance_embeds}")

        print(f"✅ DIT Transformer 로드 완료: {self.model_name}")

    def _get_parameter(self, key: str, default: str = "") -> str:
        """모델 파라미터 추출"""
        for param in self.model_config.get('parameters', []):
            if param['key'] == key:
                return param['value']['string_value']
        return default

    def execute(self, requests: List) -> List:
        """배치 추론 실행"""
        responses = []

        for request in requests:
            try:
                response = self._process_request(request)
                responses.append(response)
            except Exception as e:
                error_msg = f"DIT Transformer 처리 중 오류: {str(e)}"
                print(f"❌ {error_msg}")

                # 에러 응답 생성
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_msg)
                ) if TRITON_AVAILABLE else None
                responses.append(error_response)

        return responses

    def _process_request(self, request):
        """단일 요청 처리"""
        if TRITON_AVAILABLE:
            # Triton 모드: 실제 데이터
            hidden_states = self._get_input_tensor(request, "hidden_states")
            timestep = self._get_input_tensor(request, "timestep")
            guidance = self._get_input_tensor(request, "guidance")
            pooled_projections = self._get_input_tensor(request, "pooled_projections")
            encoder_hidden_states = self._get_input_tensor(request, "encoder_hidden_states")
            txt_ids = self._get_input_tensor(request, "txt_ids")
            img_ids = self._get_input_tensor(request, "img_ids")
        else:
            # 테스트 모드: 더미 데이터
            batch_size = 1
            num_patches = 2304  # 실제 FLUX에서 관찰된 패치 수
            hidden_states = torch.randn((batch_size, num_patches, 64), dtype=torch.bfloat16, device=self.device)
            timestep = torch.tensor([0.5], dtype=torch.bfloat16, device=self.device)
            # guidance 처리 (flux_pipeline.py 로직 기반)
            guidance_scale = 0.0
            if hasattr(self, 'guidance_embeds') and self.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=self.device, dtype=torch.float32)
                guidance = guidance.expand(batch_size)
            else:
                guidance = None
            pooled_projections = torch.randn((batch_size, 768), dtype=torch.bfloat16, device=self.device)
            encoder_hidden_states = torch.randn((batch_size, 512, 4096), dtype=torch.bfloat16, device=self.device)
            txt_ids = torch.zeros((512, 3), dtype=torch.bfloat16, device=self.device)
            img_ids = torch.zeros((num_patches, 3), dtype=torch.bfloat16, device=self.device)

        # DIT Transformer 추론 수행 (flux_pipeline.py 로직 기반)
        noise_pred = self._predict_noise(
            hidden_states, timestep, guidance, pooled_projections,
            encoder_hidden_states, txt_ids, img_ids
        )

        # 응답 생성
        if TRITON_AVAILABLE:
            output_tensor = self._torch_to_tensor(noise_pred, "noise_pred")
            return pb_utils.InferenceResponse([output_tensor])
        else:
            return noise_pred

    def _get_input_tensor(self, request, name: str) -> torch.Tensor:
        """입력 텐서 추출 및 변환"""
        triton_tensor = pb_utils.get_input_tensor_by_name(request, name)
        return self._tensor_to_torch(triton_tensor)

    def _tensor_to_torch(self, triton_tensor) -> torch.Tensor:
        """Triton 텐서를 PyTorch 텐서로 변환"""
        if DLPACK_AVAILABLE and hasattr(triton_tensor, 'to_dlpack'):
            # DLPack 사용 (메모리 복사 최소화)
            return torch.utils.dlpack.from_dlpack(triton_tensor.to_dlpack())
        else:
            # Fallback: numpy를 통한 변환
            numpy_array = triton_tensor.as_numpy()
            tensor = torch.from_numpy(numpy_array).to(self.device)
            # bf16으로 변환 (메모리 최적화)
            return tensor.to(dtype=torch.bfloat16)

    def _torch_to_tensor(self, torch_tensor: torch.Tensor, name: str):
        """PyTorch 텐서를 Triton 텐서로 변환"""
        if DLPACK_AVAILABLE:
            # DLPack 사용
            return pb_utils.Tensor.from_dlpack(name, torch.utils.dlpack.to_dlpack(torch_tensor))
        else:
            # Fallback: numpy를 통한 변환
            numpy_array = torch_tensor.detach().cpu().numpy()
            return pb_utils.Tensor(name, numpy_array)

    def _predict_noise(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        pooled_projections: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        DIT Transformer를 사용한 노이즈 예측

        flux_pipeline.py의 transformer 호출 로직 기반 (lines 944-954)

        Args:
            hidden_states: (batch_size, num_patches, 64) - 현재 latent states
            timestep: (batch_size,) - 정규화된 timestep (0-1)
            guidance: (batch_size,) - guidance scale
            pooled_projections: (batch_size, 768) - CLIP pooled embeddings
            encoder_hidden_states: (batch_size, 512, 4096) - T5 sequence embeddings
            txt_ids: (batch_size, 512, 3) - 텍스트 위치 ID
            img_ids: (batch_size, num_patches, 3) - 이미지 위치 ID

        Returns:
            noise_pred: (batch_size, num_patches, 64) - 예측된 노이즈
        """
        # 입력 텐서들을 적절한 디바이스와 dtype으로 변환
        hidden_states = hidden_states.to(device=self.device, dtype=self.transformer.dtype)
        timestep = timestep.to(device=self.device, dtype=self.transformer.dtype)
        # guidance는 None일 수 있으므로 체크
        if guidance is not None:
            guidance = guidance.to(device=self.device, dtype=self.transformer.dtype)
        pooled_projections = pooled_projections.to(device=self.device, dtype=self.transformer.dtype)
        encoder_hidden_states = encoder_hidden_states.to(device=self.device, dtype=self.transformer.dtype)
        txt_ids = txt_ids.to(device=self.device, dtype=self.transformer.dtype)
        img_ids = img_ids.to(device=self.device, dtype=self.transformer.dtype)

        # 배치 크기 확장 (flux_pipeline.py line 941 기반)
        batch_size = hidden_states.shape[0]
        if timestep.dim() == 1 and timestep.shape[0] == 1:
            timestep = timestep.expand(batch_size)
        # guidance가 None이 아닌 경우만 확장
        if guidance is not None and guidance.dim() == 1 and guidance.shape[0] == 1:
            guidance = guidance.expand(batch_size)

        print(f"📊 DIT 입력 형태:")
        print(f"  hidden_states: {hidden_states.shape}")  # [batch_size, num_patches, 64]
        print(f"  timestep: {timestep.shape}")  # [batch_size]
        print(f"  guidance: {guidance.shape if guidance is not None else 'None'}")  # [batch_size] or None
        print(f"  pooled_projections: {pooled_projections.shape}")  # [batch_size, 768]
        print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")  # [batch_size, 512, 4096]

        with torch.no_grad():
            # DIT Transformer 실행 (flux_pipeline.py lines 944-954 기반)
            # guidance는 None이라도 항상 전달 (transformer.config.guidance_embeds에 따라 처리됨)
            transformer_output = self.transformer(
                hidden_states=hidden_states,
                timestep=timestep / 1000,  # timestep 정규화
                guidance=guidance,  # None이라도 항상 전달
                pooled_projections=pooled_projections,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )

            # 노이즈 예측 추출 (첫 번째 출력)
            noise_pred = transformer_output[0]

            print(f"📊 DIT 출력 형태: {noise_pred.shape}")  # [batch_size, num_patches, 64]

            return noise_pred

    def finalize(self) -> None:
        """모델 정리"""
        print("🔄 DIT Transformer 정리 중...")
        if hasattr(self, 'transformer'):
            del self.transformer

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ DIT Transformer 정리 완료")


def test_dit_transformer():
    """직접 실행 테스트 (헌장 요구사항)"""
    print("🚀 DIT Transformer 테스트 모드")
    print("=" * 50)

    # 테스트 환경 설정
    os.environ["TEST_MODE"] = "true"

    # 모델 초기화
    model = TritonPythonModel()
    args = {
        'model_config': json.dumps({
            'parameters': [
                {'key': 'model_name', 'value': {'string_value': 'black-forest-labs/FLUX.1-schnell'}},
                {'key': 'in_channels', 'value': {'string_value': '64'}},
                {'key': 'num_inference_steps', 'value': {'string_value': '4'}},
                {'key': 'use_fp8', 'value': {'string_value': 'true'}}
            ]
        })
    }

    try:
        model.initialize(args)

        # 테스트 데이터 생성 (4회 반복 호출을 위한 설계)
        batch_size = 1
        num_patches = 2304  # 실제 FLUX에서 관찰된 패치 수

        print(f"📋 테스트 입력 형태:")
        print(f"  batch_size: {batch_size}")
        print(f"  num_patches: {num_patches}")

        # 4회 반복 시뮬레이션 (denoising steps)
        for step in range(4):
            print(f"\n🔄 Step {step + 1}/4")

            # 테스트 텐서 생성
            hidden_states = torch.randn((batch_size, num_patches, 64), dtype=torch.bfloat16)
            timestep = torch.tensor([1.0 - step * 0.25], dtype=torch.bfloat16)  # 1.0 -> 0.0

            # guidance 처리 (transformer.config.guidance_embeds에 따라)
            guidance_scale = 0.0
            if hasattr(model, 'guidance_embeds') and model.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=model.device, dtype=torch.float32)
                guidance = guidance.expand(batch_size)
            else:
                guidance = None
            pooled_projections = torch.randn((batch_size, 768), dtype=torch.bfloat16)
            encoder_hidden_states = torch.randn((batch_size, 512, 4096), dtype=torch.bfloat16)
            txt_ids = torch.zeros((512, 3), dtype=torch.bfloat16)
            img_ids = torch.zeros((num_patches, 3), dtype=torch.bfloat16)

            # 추론 실행
            noise_pred = model._predict_noise(
                hidden_states, timestep, guidance, pooled_projections,
                encoder_hidden_states, txt_ids, img_ids
            )

            # 결과 검증
            expected_shape = (batch_size, num_patches, 64)
            if noise_pred.shape == expected_shape:
                print(f"✅ Step {step + 1} 출력 형태 검증 통과: {noise_pred.shape}")
            else:
                print(f"❌ Step {step + 1} 출력 형태 오류: {noise_pred.shape}, 예상: {expected_shape}")
                return False

            # 값 범위 검증
            min_val, max_val = noise_pred.min().item(), noise_pred.max().item()
            print(f"📊 Step {step + 1} 출력 값 범위: {min_val:.4f} ~ {max_val:.4f}")

        model.finalize()
        print("\n🎉 DIT Transformer 4회 반복 테스트 성공!")
        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    # 직접 실행 시 테스트 모드
    success = test_dit_transformer()
    sys.exit(0 if success else 1)