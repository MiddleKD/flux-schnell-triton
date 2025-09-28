# Research: FLUX Pipeline to Triton Conversion

## Key Findings from flux_pipeline.py Analysis

### FLUX Pipeline Architecture Components

**Decision**: 5개의 주요 Triton 모델로 분해
**Rationale**: flux_pipeline.py의 __call__ 메서드 실행 흐름을 기반으로 분석
**Alternatives considered**: 단일 모델 vs 세분화된 분해 (5개 모델이 최적)

#### 1. BLS Orchestrator (CPU)
- **Location in Original**: FluxPipeline.__call__ 메서드 전체 흐름
- **Responsibility**:
  - Input validation (check_inputs)
  - Batch management
  - Model간 호출 순서 제어
  - 에러 처리 및 복구
- **Tensor Flow**: text → clip_embeds, t5_embeds → latents → image

#### 2. CLIP Text Encoder (GPU)
- **Location in Original**: _get_clip_prompt_embeds 메서드 (lines 267-309)
- **Responsibility**:
  - CLIP tokenization 및 encoding
  - Pooled output 생성
- **Input**: text prompts (batch_size, max_length=77)
- **Output**: pooled_prompt_embeds (batch_size * num_images_per_prompt, 768)

#### 3. T5 Text Encoder (GPU)
- **Location in Original**: _get_t5_prompt_embeds 메서드 (lines 218-265)
- **Responsibility**:
  - T5 tokenization 및 encoding
  - Sequence embeddings 생성
- **Input**: text prompts (batch_size, max_length=512)
- **Output**: prompt_embeds (batch_size * num_images_per_prompt, seq_len, 4096)

#### 4. DIT Transformer (GPU)
- **Location in Original**: transformer 호출 부분 (lines 944-954, denoising loop)
- **Responsibility**:
  - Noise prediction via transformer
  - 4회 반복 호출 (timesteps에 따라)
  - Guidance 및 attention 처리
- **Input**: latents, timestep, embeddings, guidance
- **Output**: noise_pred (latents shape와 동일)

#### 5. VAE Decoder (GPU)
- **Location in Original**: vae.decode 호출 (lines 1004-1007)
- **Responsibility**:
  - Latent → image decoding
  - Scaling 및 post-processing
- **Input**: latents (batch_size, channels, height//8, width//8)
- **Output**: images (batch_size, 3, height, width)

### DLPack Memory Optimization Strategy

**Decision**: GPU-CPU 간 최소 메모리 복사
**Rationale**: BLS는 CPU에서 실행되지만 나머지는 GPU에서 실행
**Implementation**:
- BLS에서 입력 처리 후 GPU 모델들로 DLPack tensor 전달
- GPU 모델 간에는 메모리 복사 없이 직접 전달
- 최종 결과만 CPU로 반환

### Tensor Shape Documentation

```python
# Text inputs
text_prompts: List[str]  # Variable length strings

# CLIP outputs
pooled_prompt_embeds: (batch_size, 768) dtype=float32

# T5 outputs
prompt_embeds: (batch_size, 512, 4096) dtype=float32
text_ids: (512, 3) dtype=float32

# Latent tensors
latents: (batch_size, num_patches, 64) dtype=float32
latent_image_ids: (num_patches, 3) dtype=float32

# Image outputs
images: (batch_size, 3, height, width) dtype=float32
```

### Configuration Templates Required

**Decision**: 각 모델별 config.pbtxt 템플릿 생성
**Rationale**: Triton 표준 구조 준수 및 배치 처리 지원

1. **BLS config**: CPU 실행, 동적 배치 크기
2. **CLIP config**: GPU 실행, 고정/가변 시퀀스 길이
3. **T5 config**: GPU 실행, 최대 512 토큰
4. **DIT config**: GPU 실행, 동적 latent 크기
5. **VAE config**: GPU 실행, 동적 이미지 크기

### Test Strategy from flux_pipeline.py

**Decision**: 기존 파이프라인과 동등성 검증
**Rationale**: 각 모델의 출력이 원본과 일치해야 함

1. **단위 테스트**: 각 model.py 직접 실행
2. **통합 테스트**: 전체 파이프라인 비교
3. **성능 테스트**: 메모리 사용량 및 속도

### Implementation Exclusions (from CLAUDE.md)

- negative_prompt/do_true_cfg: 미지원 (lines 823, 956-972 제외)
- IP adapters: 미지원 (lines 896-926 제외)
- LoRA: 미지원 (lines 347-354, 376-383 제외)
- 입력은 text prompts만 처리

### Key Dependencies Analysis

**Decision**: 핵심 의존성 최소화
**Rationale**: 템플릿 중심 접근법으로 복잡성 감소

- **필수**: torch, transformers, triton-python-backend
- **선택**: diffusers (참조용), pytest (테스트용)
- **제외**: 복잡한 스케줄러, 고급 최적화

## Next Phase Inputs

이 연구 결과는 Phase 1 Design & Contracts에서 다음과 같이 활용됩니다:

1. **Data Model**: 각 모델의 입력/출력 텐서 정의
2. **Contracts**: 5개 모델의 API 인터페이스 명세
3. **Quickstart**: flux_pipeline.py 기반 검증 시나리오