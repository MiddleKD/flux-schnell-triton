# Tasks: FLUX Pipeline to Triton Conversion

**Input**: Design documents from `/specs/002-claude-md/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Extract: tech stack (Python, PyTorch, Triton), libraries (diffusers, transformers)
   → Structure: 5 Triton models in triton_models/ directory
2. Load design documents:
   → data-model.md: TextPrompt, PromptEmbeddings, LatentTensor, GeneratedImage entities
   → contracts/: 5 model APIs (BLS, CLIP, T5, DIT, VAE)
   → research.md: DLPack optimization, flux_pipeline.py structure preservation
3. Generate tasks by category:
   → Setup: triton_models structure, dependencies, linting
   → Tests: contract tests for 5 models, integration tests
   → Core: 5 model implementations (config.pbtxt + model.py each)
   → Integration: DLPack optimization, model orchestration
   → Polish: performance tests, documentation, validation
4. Apply task rules:
   → Different models = [P] parallel implementation
   → Same file modifications = sequential
   → Tests before implementation (TDD)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Triton models**: `triton_models/[model_name]/` structure
- **Tests**: `tests/` at repository root
- **Source**: Direct in `triton_models/[model_name]/1/model.py`

## Phase 3.1: Setup
- [x] T001 Create triton_models directory structure for 5 models (bls, clip_encoder, t5_encoder, dit_transformer, vae_decoder)
- [x] T002 Initialize Python environment with torch, transformers, triton-python-backend dependencies using uv

## Phase 3.2: Basic Tests
- [x] T003 [P] Basic test for individual model execution via model.py direct run
- [x] T004 [P] Integration test single prompt scenario in tests/integration/test_single_prompt.py
- [x] T005 [P] Integration test batch processing scenario in tests/integration/test_batch_processing.py

## Phase 3.3: Core Implementation
- [x] T006 [P] BLS orchestrator config.pbtxt in triton_models/bls/config.pbtxt
- [x] T007 [P] CLIP encoder config.pbtxt in triton_models/clip_encoder/config.pbtxt
- [x] T008 [P] T5 encoder config.pbtxt in triton_models/t5_encoder/config.pbtxt
- [x] T009 [P] DIT transformer config.pbtxt in triton_models/dit_transformer/config.pbtxt
- [x] T010 [P] VAE decoder config.pbtxt in triton_models/vae_decoder/config.pbtxt
- [x] T011 [P] CLIP encoder model.py in triton_models/clip_encoder/1/model.py
- [x] T012 [P] T5 encoder model.py in triton_models/t5_encoder/1/model.py
- [x] T013 [P] DIT transformer model.py in triton_models/dit_transformer/1/model.py
- [x] T014 [P] VAE decoder model.py in triton_models/vae_decoder/1/model.py
- [x] T015 BLS orchestrator model.py in triton_models/bls/1/model.py

## Phase 3.4: Integration
- [x] T016 Implement timestep loop logic in BLS for DIT transformer calls
- [x] T017 Basic tensor shape validation and batch size handling

## 구현 완료 상태

✅ **모든 핵심 컴포넌트 구현 완료:**
- **BLS Orchestrator**: CPU에서 실행되는 메인 오케스트레이터
- **CLIP Encoder**: text_encoder subfolder 사용, 성공적 테스트 완료
- **T5 Encoder**: text_encoder_2/tokenizer_2 subfolder 사용, 성공적 테스트 완료
- **DIT Transformer**: transformer subfolder, 올바른 guidance 처리 로직 구현
- **VAE Decoder**: vae subfolder 사용

✅ **FLUX.1-schnell 구조 준수:**
- 모든 모델이 올바른 subfolder 사용
- flux_pipeline.py 로직 보존
- model_index.json 구조 반영

✅ **헌장 준수:**
- 단순성 우선 (fallback 로직 제거)
- 500줄 이하 파일 크기
- TDD 지원 (직접 실행 가능)
- 우아한 성능 저하 (DLPack 분기처리)

**참고**: DIT transformer는 메모리 제약으로 실제 테스트 불가하지만 구조와 로직은 완전히 구현됨

## Dependencies
- Setup (T001-T002) before tests (T003-T005)
- Tests (T003-T005) before implementation (T006-T015)
- Config files (T006-T010) before model implementations (T011-T015)
- Core implementation (T006-T015) before integration (T016-T017)
- T015 (BLS) depends on T011-T014 (other models) for orchestration logic

## Parallel Example
```
# Launch config files together (T006-T010):
Task: "BLS orchestrator config.pbtxt in triton_models/bls/config.pbtxt"
Task: "CLIP encoder config.pbtxt in triton_models/clip_encoder/config.pbtxt"
Task: "T5 encoder config.pbtxt in triton_models/t5_encoder/config.pbtxt"
Task: "DIT transformer config.pbtxt in triton_models/dit_transformer/config.pbtxt"
Task: "VAE decoder config.pbtxt in triton_models/vae_decoder/config.pbtxt"

# Launch model implementations together (T011-T014, excluding BLS):
Task: "CLIP encoder model.py in triton_models/clip_encoder/1/model.py"
Task: "T5 encoder model.py in triton_models/t5_encoder/1/model.py"
Task: "DIT transformer model.py in triton_models/dit_transformer/1/model.py"
Task: "VAE decoder model.py in triton_models/vae_decoder/1/model.py"
```

## Notes
- [P] tasks = different files, no dependencies between them
- Reference flux_pipeline.py for tensor shapes and processing logic
- Each model.py should be directly executable for testing
- Maintain tensor shape comments as specified in data-model.md
- BLS orchestrator coordinates the entire pipeline flow
- GPU models (CLIP, T5, DIT, VAE) vs CPU model (BLS)
- Support batch processing for multiple prompts simultaneously

## Validation Checklist
*Essential functionality only*

- [x] Core 5 models have config and implementation tasks
- [x] Basic tests for model execution
- [x] Batch processing functionality included
- [x] Parallel tasks truly independent (different files)
- [x] Each task specifies exact file path
- [x] BLS depends on other models for orchestration
- [x] Triton model structure properly organized