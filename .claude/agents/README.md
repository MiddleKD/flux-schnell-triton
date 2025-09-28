# FLUX to Triton Sub-Agents

FLUX Pipeline을 Triton Inference Server로 변환하는 작업을 병렬 실행하기 위한 전문화된 sub-agent 설정 파일들입니다.

## Agent 구성

### 1. triton-config-agent.md
**역할**: Triton 설정 파일(.pbtxt) 생성
**담당**: T006-T010 (5개 모델의 config.pbtxt)
**전문성**: Triton 서버 설정, 동적 배치, GPU/CPU 최적화

### 2. text-encoder-agent.md
**역할**: CLIP 및 T5 텍스트 인코더 구현
**담당**: T011-T012 (CLIP, T5 model.py)
**전문성**: Transformers 라이브러리, 텍스트 임베딩, 배치 처리

### 3. dit-transformer-agent.md
**역할**: DIT 트랜스포머 모델 구현
**담당**: T013 (DIT model.py)
**전문성**: Diffusion 모델, 노이즈 예측, 타임스텝 처리

### 4. vae-decoder-agent.md
**역할**: VAE 디코더 모델 구현
**담당**: T014 (VAE model.py)
**전문성**: 이미지 디코딩, 텐서 변환, 후처리

### 5. bls-orchestrator-agent.md
**역할**: BLS 오케스트레이터 구현
**담당**: T015 (BLS model.py)
**전문성**: 파이프라인 조정, 비동기 호출, 에러 처리

### 6. testing-agent.md
**역할**: 테스트 및 검증
**담당**: T003-T005 (테스트 시나리오)
**전문성**: 통합 테스트, 성능 벤치마킹, 검증

## 병렬 실행 가능한 작업 그룹

### Phase 1: Config 파일 생성 (동시 실행 가능)
```bash
# 모든 config.pbtxt 파일을 병렬로 생성
Task: triton-config-agent T006  # BLS config
Task: triton-config-agent T007  # CLIP config
Task: triton-config-agent T008  # T5 config
Task: triton-config-agent T009  # DIT config
Task: triton-config-agent T010  # VAE config
```

### Phase 2: 모델 구현 (부분 병렬)
```bash
# 텍스트 인코더들 병렬 실행
Task: text-encoder-agent T011   # CLIP model.py
Task: text-encoder-agent T012   # T5 model.py

# DIT와 VAE 병렬 실행
Task: dit-transformer-agent T013  # DIT model.py
Task: vae-decoder-agent T014     # VAE model.py

# BLS는 다른 모델들 완료 후 실행
Task: bls-orchestrator-agent T015  # BLS model.py (의존성 있음)
```

### Phase 3: 테스트 (병렬 실행 가능)
```bash
# 모든 테스트 동시 실행
Task: testing-agent T003  # 개별 모델 테스트
Task: testing-agent T004  # 단일 프롬프트 테스트
Task: testing-agent T005  # 배치 처리 테스트
```

## 공통 원칙

### flux_pipeline.py 필수 참조
모든 agent는 `@flux_pipeline.py`를 핵심 참조 자료로 활용해야 합니다:
- 텐서 형태 및 데이터 타입
- 처리 로직 및 순서
- 에러 처리 방식
- 성능 최적화 패턴

### 배치 처리 지원
모든 모델은 배치 처리를 지원해야 합니다:
- 단일 프롬프트: `batch_size=1`
- 다중 프롬프트: `batch_size=N`
- 동적 배치 크기 처리

### 직접 실행 가능
각 model.py는 독립적으로 테스트 가능해야 합니다:
```bash
# uv run을 사용한 직접 실행
uv run triton_models/clip_encoder/1/model.py
uv run triton_models/t5_encoder/1/model.py
uv run triton_models/dit_transformer/1/model.py
uv run triton_models/vae_decoder/1/model.py
uv run triton_models/bls/1/model.py
```

```python
if __name__ == "__main__":
    # 직접 테스트 실행 코드
    test_model_functionality()
```

### 에러 처리
- 입력 검증 및 명확한 에러 메시지
- GPU 메모리 부족 상황 처리
- 모델 로딩 실패 복구 로직

## 사용법

1. **Agent 선택**: 작업에 맞는 agent 선택
2. **컨텍스트 제공**: @flux_pipeline.py 참조 필수
3. **병렬 실행**: 의존성 없는 작업들 동시 실행
4. **검증**: testing-agent로 결과 검증

## 성공 기준

- **동등성**: flux_pipeline.py와 동일한 결과
- **성능**: 원본 대비 동등하거나 향상된 성능
- **확장성**: 배치 처리로 처리량 향상
- **안정성**: 에러 상황 적절한 처리