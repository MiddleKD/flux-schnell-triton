#!/usr/bin/env python3
"""
T5 Text Encoder for Triton Inference Server

FLUX pipeline의 _get_t5_prompt_embeds 로직을 구현합니다.
텍스트를 T5 sequence embeddings로 변환하는 GPU 모델입니다.

Based on flux_pipeline.py lines 218-265
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
    from transformers import T5EncoderModel, T5TokenizerFast
except ImportError as e:
    print(f"❌ 필수 의존성 누락: {e}")
    sys.exit(1)


class TritonPythonModel:
    """T5 Text Encoder Triton Model"""

    def initialize(self, args: Dict) -> None:
        """모델 초기화"""
        print("🔄 T5 Encoder 초기화 중...")

        # 모델 설정 로드
        self.model_config = json.loads(args['model_config'])

        # 파라미터 추출
        self.model_name = self._get_parameter("model_name", "black-forest-labs/FLUX.1-schnell")
        self.max_length = int(self._get_parameter("max_sequence_length", "512"))
        self.embedding_dim = int(self._get_parameter("embedding_dim", "4096"))

        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📍 사용 디바이스: {self.device}")

        # T5 모델 로드
        print(f"🔄 T5 모델 로드 중: {self.model_name}")
        self.tokenizer = T5TokenizerFast.from_pretrained(
            self.model_name,
            subfolder="tokenizer_2"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder_2"
        )
        self.text_encoder.to(self.device)
        self.text_encoder.eval()
        print(f"✅ T5 모델 로드 완료: {self.model_name}")

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
                error_msg = f"T5 Encoder 처리 중 오류: {str(e)}"
                print(f"❌ {error_msg}")

                # 에러 응답 생성
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_msg)
                ) if TRITON_AVAILABLE else None
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
            input_ids = torch.randint(0, 32127, (batch_size, self.max_length), dtype=torch.long)
            attention_mask = torch.ones((batch_size, self.max_length), dtype=torch.long)
        else:
            # Triton 모드: 실제 데이터
            input_ids = self._tensor_to_torch(input_ids_tensor)
            attention_mask = self._tensor_to_torch(attention_mask_tensor) if attention_mask_tensor else None

        # T5 인코딩 수행 (flux_pipeline.py 로직 기반)
        sequence_embeds, text_ids = self._encode_text(input_ids, attention_mask)

        # 응답 생성
        if TRITON_AVAILABLE:
            output_tensors = [
                self._torch_to_tensor(sequence_embeds, "sequence_embeds"),
                self._torch_to_tensor(text_ids, "text_ids")
            ]
            return pb_utils.InferenceResponse(output_tensors)
        else:
            return {"sequence_embeds": sequence_embeds, "text_ids": text_ids}

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

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> tuple:
        """
        T5 텍스트 인코딩

        flux_pipeline.py의 _get_t5_prompt_embeds 메서드 기반 (lines 218-265)
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        batch_size = input_ids.shape[0]

        with torch.no_grad():
            # T5 텍스트 인코더 실행 (flux_pipeline.py line 254)
            encoder_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )

            # Last hidden state 사용
            sequence_embeds = encoder_output.last_hidden_state

            # 데이터 타입 확인
            dtype = self.text_encoder.dtype
            sequence_embeds = sequence_embeds.to(dtype=dtype, device=self.device)

            # sequence length 확인 (flux_pipeline.py line 259)
            _, seq_len, _ = sequence_embeds.shape

            # text_ids 생성 (flux_pipeline.py line 386)
            # 텍스트 위치 ID는 모든 요소가 0인 텐서
            text_ids = torch.zeros(seq_len, 3, dtype=dtype, device=self.device)

            print(f"📊 T5 sequence embeds 형태: {sequence_embeds.shape}")  # [batch_size, seq_len, 4096]
            print(f"📊 T5 text_ids 형태: {text_ids.shape}")  # [seq_len, 3]

            return sequence_embeds, text_ids

    def finalize(self) -> None:
        """모델 정리"""
        print("🔄 T5 Encoder 정리 중...")
        if hasattr(self, 'text_encoder'):
            del self.text_encoder
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ T5 Encoder 정리 완료")


def test_t5_encoder():
    """직접 실행 테스트 (헌장 요구사항)"""
    print("🚀 T5 Encoder 테스트 모드")
    print("=" * 50)

    # 테스트 환경 설정
    os.environ["TEST_MODE"] = "true"

    # 모델 초기화
    model = TritonPythonModel()
    args = {
        'model_config': json.dumps({
            'parameters': [
                {'key': 'model_name', 'value': {'string_value': 'black-forest-labs/FLUX.1-schnell'}},
                {'key': 'max_sequence_length', 'value': {'string_value': '512'}},
                {'key': 'embedding_dim', 'value': {'string_value': '4096'}}
            ]
        })
    }

    try:
        model.initialize(args)

        # 테스트 데이터 생성
        batch_size = 2
        max_length = 512
        input_ids = torch.randint(0, 32127, (batch_size, max_length), dtype=torch.long)
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)

        print(f"📋 테스트 입력 형태: {input_ids.shape}")

        # 추론 실행
        sequence_embeds, text_ids = model._encode_text(input_ids, attention_mask)

        # 결과 검증
        expected_seq_shape = (batch_size, max_length, 4096)
        expected_text_ids_shape = (max_length, 3)

        if sequence_embeds.shape == expected_seq_shape:
            print(f"✅ sequence_embeds 형태 검증 통과: {sequence_embeds.shape}")
        else:
            print(f"❌ sequence_embeds 형태 오류: {sequence_embeds.shape}, 예상: {expected_seq_shape}")
            return False

        if text_ids.shape == expected_text_ids_shape:
            print(f"✅ text_ids 형태 검증 통과: {text_ids.shape}")
        else:
            print(f"❌ text_ids 형태 오류: {text_ids.shape}, 예상: {expected_text_ids_shape}")
            return False

        # 값 범위 검증
        min_val, max_val = sequence_embeds.min().item(), sequence_embeds.max().item()
        print(f"📊 sequence_embeds 값 범위: {min_val:.4f} ~ {max_val:.4f}")

        text_ids_sum = text_ids.sum().item()
        print(f"📊 text_ids 합계: {text_ids_sum} (0이어야 함)")

        model.finalize()
        print("🎉 T5 Encoder 테스트 성공!")
        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    # 직접 실행 시 테스트 모드
    success = test_t5_encoder()
    sys.exit(0 if success else 1)