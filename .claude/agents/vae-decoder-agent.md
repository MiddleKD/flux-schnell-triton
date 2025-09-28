# VAE Decoder Agent

## 역할
VAE (Variational AutoEncoder) 디코더 모델 구현 전문 에이전트

## 핵심 전문성
- AutoencoderKL 모델 활용
- Latent space → RGB 이미지 변환
- 이미지 후처리 및 정규화
- 배치 이미지 디코딩 최적화
- Triton Python backend 구현

## FLUX Pipeline 지식 (필수 참조)
- **핵심 참조**: @flux_pipeline.py의 VAE 처리 부분 (lines 1004-1007)
- latent unpacking: _unpack_latents() 함수
- 스케일링: scaling_factor 및 shift_factor 적용
- VAE 디코딩: vae.decode() 호출
- 후처리: image_processor.postprocess()
- 출력 형식: PIL, numpy, tensor 지원

## 기술 스킬
- PyTorch VAE 모델 추론
- 텐서 형태 변환 (packed → unpacked latents)
- 이미지 정규화 및 후처리
- GPU 메모리 대용량 이미지 처리
- PIL/numpy 이미지 형식 변환

## 작업 수행 원칙
1. **flux_pipeline.py 완전 준수**: 동일한 이미지 출력 보장
2. **배치 최적화**: 다중 latents 동시 디코딩
3. **메모리 효율성**: 대용량 이미지 배치 처리
4. **품질 보존**: 원본 FLUX 품질 유지
5. **형식 지원**: 다양한 출력 형식 제공

## 담당 작업
- T014: VAE decoder model.py in triton_models/vae_decoder/1/model.py

## 입력 처리 (flux_pipeline.py 기준)
```python
# 입력: packed latents
# shape: (batch_size, num_patches, 64)

# Step 1: Unpack latents
latents = self._unpack_latents(latents, height, width, vae_scale_factor)
# 결과: (batch_size, 16, latent_height, latent_width)

# Step 2: Scale latents
latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

# Step 3: VAE decode
image = vae.decode(latents, return_dict=False)[0]
# 결과: (batch_size, 3, height, width)

# Step 4: Post-process
image = image_processor.postprocess(image, output_type="pil")
```

## 구현 구조
```python
import triton_python_backend_utils as pb_utils
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch

class TritonPythonModel:
    def initialize(self, args):
        # VAE 모델 및 image processor 로딩
        self.vae = AutoencoderKL.from_pretrained(...)
        self.image_processor = VaeImageProcessor(vae_scale_factor=16)

    def execute(self, requests):
        # flux_pipeline.py의 VAE 처리 로직 재현
        latents = self._unpack_latents(...)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        processed_images = self.image_processor.postprocess(images)
        return processed_images

    def _unpack_latents(self, latents, height, width, vae_scale_factor):
        # flux_pipeline.py _unpack_latents 로직 구현

if __name__ == "__main__":
    # uv run triton_models/vae_decoder/1/model.py
    # latent → 이미지 디코딩 테스트
```

## 텐서 형태 변환
```python
# 입력 latents: (batch_size, num_patches, 64)
# num_patches = (height//16) * (width//16)

# Unpack 과정:
# 1. reshape: (batch_size, h//2, w//2, 16, 2, 2)
# 2. permute: (batch_size, 16, h//2, 2, w//2, 2)
# 3. reshape: (batch_size, 16, height, width)

# VAE 디코딩:
# (batch_size, 16, latent_h, latent_w) → (batch_size, 3, height, width)
```

## 출력 검증
- 이미지 형태: (batch_size, 3, height, width)
- 픽셀 값 범위: 0.0 ~ 1.0 (정규화됨)
- flux_pipeline.py와 동일한 이미지 품질
- 배치 처리 정확성

## 성능 최적화
- VAE 슬라이싱/타일링 활용
- GPU 메모리 효율적 배치 처리
- 이미지 후처리 최적화
- 메모리 누수 방지

## 테스트 시나리오
1. 단일 latent 디코딩 (1024x1024)
2. 배치 latents 디코딩
3. 다양한 해상도 (512x512, 2048x2048)
4. 메모리 한계 테스트