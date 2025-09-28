<!--
Sync Impact Report:
- Version change: Initial → 1.0.0
- Initial constitution creation based on CLAUDE.md
- Added principles: Test-First Development, Simplicity First, Graceful Degradation, File Size Discipline, Template-Driven Development
- Added sections: Technology Stack, Development Workflow
- Templates requiring updates: ✅ none (initial creation)
- Follow-up TODOs: none
-->

# FLUX Triton Pipeline Constitution

## Core Principles

### I. Test-First Development (NON-NEGOTIABLE)
TDD는 필수입니다: 테스트를 먼저 작성하고 사용자 승인을 받은 후 구현합니다.
1순위 테스트 방식은 `model.py`를 직접 실행하는 것이며, 2순위로 pytest를 사용합니다.
테스트는 너무 많지 않게 정말 필요한 부분만 작성하되, Red-Green-Refactor 사이클을 엄격히 준수해야 합니다.

**근거**: 품질 보장과 안정적인 FLUX to Triton 변환을 위해 테스트 우선 개발이 필수적입니다.

### II. Simplicity First
단순성을 우선시하며 복잡한 패턴을 금지합니다.
완벽히 동작할 필요는 없으며 템플릿을 작성하는 것을 우선으로 합니다.
YAGNI(You Aren't Gonna Need It) 원칙을 철저히 따라야 합니다.

**근거**: 복잡성은 버그와 유지보수 비용을 증가시키며, 템플릿 기반 접근법이 프로젝트 목표에 부합합니다.

### III. Graceful Degradation
누락된 의존성이 있을 때 기능을 비활성화하되 전체 시스템은 계속 동작해야 합니다.
DLPack 사용 시 테스트 환경에서는 분기 처리를 통해 우아하게 처리해야 합니다.

**근거**: 다양한 환경에서의 안정성과 개발 편의성을 보장합니다.

### IV. File Size Discipline
모든 파일은 500줄 이하로 유지해야 합니다.
코드의 가독성과 유지보수성을 위해 적절한 모듈 분리가 필요합니다.

**근거**: 작은 파일은 이해하기 쉽고 테스트하기 쉬우며 버그 발생률을 낮춥니다.

### V. Template-Driven Development
각 Triton 모델에 대한 `config.pbtxt`와 `model.py` 템플릿을 우선 작성합니다.
기존 `flux_pipeline.py` 예시의 구조를 유지하려고 노력해야 합니다.
모델 간 tensor의 dtype과 shape을 고려하고 예시 shape을 명시해야 합니다.

**근거**: 일관성 있는 구조와 명확한 문서화를 통해 개발 효율성을 높입니다.

## Technology Stack

### Required Components
- **언어**: Python 3.8+
- **주요 의존성**: diffusers, transformers, pytorch, triton inference server, triton python backend(DLPack)
- **CLI 프레임워크**: Click + Rich
- **빌드 시스템**: Makefile 기반 명령어 통합
- **가상환경**: uv 필수 사용

### Architecture Requirements
Triton Pipeline 구조는 다음과 같아야 합니다:
- BLS (CPU) => CLIP => DIT * 4 => VAE
- T5 텍스트 인코더 병렬 처리
- DLPack을 이용한 GPU-CPU 메모리 복사 최소화
- 배치 추론 및 배치 단위 실행 지원

## Development Workflow

### Implementation Scope
**포함**: 텍스트 입력만 처리, 배치 추론, config.pbtxt와 model.py 작성
**제외**: negative prompt(do_true_cfg), IP adapter, LoRA

### Testing Protocol
1. `model.py` 직접 실행을 통한 1차 테스트
2. DLPack 사용 시 테스트 환경에서 분기 처리
3. 선택적으로 pytest 활용

### Documentation Standards
모든 tensor에 대해 dtype과 shape 예시를 명시해야 합니다.
예: `# some_embeds (batch_size, 768)`

## Governance

이 헌장은 모든 다른 관행들보다 우선됩니다.
모든 PR과 코드 리뷰는 헌장 준수를 검증해야 합니다.
복잡성은 반드시 정당화되어야 하며, 단순한 해결책이 우선됩니다.
헌장 수정은 문서화, 승인, 마이그레이션 계획이 필요합니다.

**Version**: 1.0.0 | **Ratified**: 2025-09-27 | **Last Amended**: 2025-09-27