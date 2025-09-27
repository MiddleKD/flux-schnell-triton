인터넷 검색이 필요할 경우 우선적으로 mcp-server-fetch 도구를 사용하세요.

# 개발 가이드

## 기술 스택
- **언어**: Python 3.8+
- **주요 의존성**: fvcore, ptflops, torch.profiler, onnxruntime, tensorrt
- **CLI 프레임워크**: Click + Rich
- **테스트**: pytest, pytest-cov
- **빌드**: Makefile 기반 명령어 통합
- **가상환경**: uv를 이용하세요

## 아키텍처 원칙
- 어댑터 패턴으로 각 프레임워크별 도구 통합 (fvcore, ptflops, onnxruntime, trtexec)
- 계층화된 의존성: 핵심/선택적 패키지로 분리
- 우아한 성능 저하: 누락된 의존성 시 기능 비활성화

## 개발 가이드라인
- TDD 필수: 테스트 먼저 작성
- 파일당 500줄 이하 유지
- 단순성 우선: 복잡한 패턴 금지

# 개발 내용

## FLUX pipeline을 Triton으로 변환

### Triton Pipeline
bls => clip => dit * 4 => vae
    => t5
(bls는 cpu, 나머지는 gpu)

#### BLS 로직
1. async_exec
    - clip text encoder
    - t5 text encoder
2. sync_exec
    - transformer(DIT) * 4
        - timesteps에 따른 4회 반복 호출
3. sync_exec
    - vae decode

### FLUX pipeline(module) 
1. text encoder
    - clip text encoder
    - t5 text encoder
2. transformer(DIT)
3. vae

### 구현 필요 내용
- batch 추론 및 batch단위 exec
- 각 모델에 대한 `config.pbtxt`와 `model.py`
- DLPack을 이용한 GPU CPU 메모리 복사 최소화

### 구현 제외 내용
- input은 오로지 text입니다.
- negative prompt(do_true_cfg)는 사용하지 않습니다.
- ip adapter는 사용하지 않습니다.
- LoRA는 사용하지 않습니다.

### 고려 사항
- 완벽히 동작할 필요는 없습니다. 템플릿을 작성하는게 우선입니다.
- 모델 간 tensor의 dtype과 shape을 고려하고 예시 shape을 작성하세요.
    - ex: # some_embeds (batch_size, 768)
- 단순성 우선: 복잡한 패턴 금지
- 테스트는 `model.py`를 실행하면 가능하도록 하되, test일 경우 DLpack 사용을 분기처리하세요.
- example `flux_pipeline.py`를 참고하고, 해당 코드의 구조를 유지하려고 노력하세요.
 