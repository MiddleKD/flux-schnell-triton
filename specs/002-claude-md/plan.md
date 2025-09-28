
# Implementation Plan: FLUX Pipeline to Triton Conversion System

**Branch**: `002-claude-md` | **Date**: 2025-09-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/home/middlek/lib/study/serving/flux_triton/specs/002-claude-md/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Convert existing FLUX pipeline from diffusers package into Triton-compatible model components. Each model.py will represent a decomposed part of the original flux_pipeline, enabling distributed inference across CPU/GPU resources with optimized memory management through DLPack. Primary focus is template creation and pipeline decomposition rather than full functionality.

## Technical Context
**Language/Version**: Python 3.8+
**Primary Dependencies**: diffusers, transformers, pytorch, triton inference server, triton python backend (DLPack), Click + Rich
**Storage**: N/A (template-focused, no persistent storage)
**Testing**: model.py direct execution (primary), pytest (secondary)
**Target Platform**: Linux server with GPU support
**Project Type**: single - Triton model library with CLI interface
**Performance Goals**: Template generation, batch inference support, memory-optimized GPU-CPU transfers
**Constraints**: 500 lines per file, simple patterns only, graceful degradation on missing dependencies
**Scale/Scope**: 5-6 Triton models (BLS, CLIP, T5, DIT*4, VAE), template generation focus

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Test-First Development (NON-NEGOTIABLE)**: ✅ PASS
- Each model.py must be directly executable for testing
- TDD workflow: model.py execution → pytest (optional)
- Tests written before implementation

**II. Simplicity First**: ✅ PASS
- Template creation prioritized over perfect functionality
- Complex patterns avoided
- YAGNI principle applied

**III. Graceful Degradation**: ✅ PASS
- DLPack usage with test environment branching
- Missing dependencies handled gracefully

**IV. File Size Discipline**: ✅ PASS
- All files kept under 500 lines
- Proper module separation for Triton models

**V. Template-Driven Development**: ✅ PASS
- config.pbtxt and model.py templates for each Triton model
- flux_pipeline.py structure preservation
- Tensor dtype/shape documentation required

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
triton_models/
├── bls/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py          # BLS orchestration (CPU)
├── clip_encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py          # CLIP text encoder (GPU)
├── t5_encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py          # T5 text encoder (GPU)
├── dit_transformer/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py          # DIT transformer (GPU, 4x calls)
└── vae_decoder/
    ├── config.pbtxt
    └── 1/
        └── model.py          # VAE decoder (GPU)

src/
├── pipeline/
│   └── flux_pipeline.py      # Original pipeline reference
├── utils/
│   ├── tensor_utils.py       # DLPack utilities
│   └── config_gen.py         # Config template generator
└── cli/
    └── convert.py            # CLI interface

tests/
├── models/                   # Direct model.py execution tests
├── integration/              # Pipeline integration tests
└── unit/                     # Utility function tests
```

**Structure Decision**: Single project with Triton model repository structure. Each model follows Triton's convention: config.pbtxt + version/model.py. BLS handles CPU orchestration while other models run on GPU with DLPack tensor passing.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh claude`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each Triton model contract → config.pbtxt + model.py creation tasks [P]
- Each API contract → contract test task [P]
- quickstart scenarios → integration test tasks
- flux_pipeline.py analysis → template generation tasks

**Specific Task Categories**:

1. **Setup Tasks**:
   - Triton model repository structure creation
   - Python package initialization with uv
   - Dependencies installation (torch, transformers, tritonclient)

2. **Template Generation Tasks** [P]:
   - BLS orchestrator config.pbtxt + model.py
   - CLIP encoder config.pbtxt + model.py
   - T5 encoder config.pbtxt + model.py
   - DIT transformer config.pbtxt + model.py
   - VAE decoder config.pbtxt + model.py

3. **Test-First Tasks** (헌장 준수):
   - Each model.py direct execution test
   - Contract validation tests
   - Quickstart scenario tests
   - Memory optimization verification tests

4. **Integration Tasks**:
   - DLPack tensor passing implementation
   - BLS orchestration logic
   - Error handling and graceful degradation
   - CLI interface creation

**Ordering Strategy**:
- TDD order: Tests before implementation (헌장 요구사항)
- Dependency order: config.pbtxt → model.py → tests
- Templates before integration (단순성 우선)
- Mark [P] for parallel execution (different model files)

**flux_pipeline.py Integration**:
- Each model.py maps to specific flux_pipeline.py methods:
  - BLS → FluxPipeline.__call__ orchestration
  - CLIP → _get_clip_prompt_embeds method
  - T5 → _get_t5_prompt_embeds method
  - DIT → transformer call in denoising loop
  - VAE → vae.decode call

**Estimated Output**: 20-25 numbered, ordered tasks in tasks.md

**Constitutional Compliance**:
- All files under 500 lines (File Size Discipline)
- model.py direct execution for testing (Test-First Development)
- Template creation over perfect functionality (Simplicity First)
- DLPack graceful degradation (Graceful Degradation)

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none required)

---
*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*
