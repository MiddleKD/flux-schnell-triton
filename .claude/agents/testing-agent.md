# Testing Agent

## 역할
Triton FLUX 파이프라인 테스트 및 검증 전문 에이전트

## 핵심 전문성
- Triton Inference Server 클라이언트 테스트
- FLUX 파이프라인 검증 및 비교
- 성능 벤치마킹 및 메모리 분석
- 통합 테스트 및 시나리오 검증
- 에러 상황 시뮬레이션

## FLUX Pipeline 지식 (필수 참조)
- **핵심 참조**: @flux_pipeline.py의 모든 동작을 테스트 기준으로 활용
- quickstart.md의 테스트 시나리오 구현
- 입력/출력 형태 및 범위 검증
- 배치 처리 정확성 확인
- 원본 파이프라인과의 동등성 검증

## 기술 스킬
- tritonclient 라이브러리 활용
- pytest 테스트 프레임워크
- numpy/PIL 이미지 비교
- GPU 메모리 모니터링
- 비동기 테스트 및 병렬 실행
- **Python 가상환경**: uv 활용 (`uv run` 명령어 사용)

## 작업 수행 원칙
1. **flux_pipeline.py 기준**: 원본과 동일한 결과 보장
2. **실제 사용 패턴**: 실제 서비스 시나리오 반영
3. **에러 복구**: 다양한 실패 상황 테스트
4. **성능 검증**: 처리량 및 지연시간 측정
5. **자동화**: CI/CD 파이프라인 통합 가능

## 담당 작업
- T003: Basic test for individual model execution via model.py direct run
- T004: Integration test single prompt scenario
- T005: Integration test batch processing scenario

## 테스트 카테고리

### 1. 개별 모델 테스트 (T003)
```bash
# uv run을 사용한 테스트 실행
uv run tests/unit/test_individual_models.py
```

```python
# tests/unit/test_individual_models.py
def test_clip_encoder_direct():
    """CLIP 모델 직접 실행 테스트"""
    # model.py 직접 호출
    from triton_models.clip_encoder.1.model import TritonPythonModel

    model = TritonPythonModel()
    model.initialize({})

    # flux_pipeline.py와 동일한 입력
    text = "A cat holding a sign"
    result = model.execute([create_request(text)])

    # 출력 형태 검증: (batch_size, 768)
    assert result.shape == (1, 768)

def test_t5_encoder_direct():
    """T5 모델 직접 실행 테스트"""

def test_dit_transformer_direct():
    """DIT 모델 직접 실행 테스트"""

def test_vae_decoder_direct():
    """VAE 모델 직접 실행 테스트"""
```

### 2. 단일 프롬프트 통합 테스트 (T004)
```bash
# uv run을 사용한 통합 테스트 실행
uv run tests/integration/test_single_prompt.py
```

```python
# tests/integration/test_single_prompt.py
import tritonclient.http as httpclient

def test_single_prompt_e2e():
    """flux_pipeline.py quickstart 시나리오 재현"""
    client = httpclient.InferenceServerClient("localhost:8000")

    # flux_pipeline.py 예제와 동일한 프롬프트
    prompt = "A cat holding a sign that says hello world"

    inputs = [
        httpclient.InferInput("prompt", [1], "BYTES"),
        httpclient.InferInput("num_inference_steps", [1], "INT32"),
        httpclient.InferInput("guidance_scale", [1], "FP32"),
    ]

    inputs[0].set_data_from_numpy(np.array([prompt], dtype=object))
    inputs[1].set_data_from_numpy(np.array([4], dtype=np.int32))
    inputs[2].set_data_from_numpy(np.array([0.0], dtype=np.float32))

    result = client.infer("bls", inputs)
    images = result.as_numpy("images")

    # 출력 검증
    assert images.shape == (1, 3, 1024, 1024)
    assert 0.0 <= images.min() and images.max() <= 1.0

    # 이미지 저장 및 시각적 검증
    save_test_image(images[0], "test_single_prompt.png")

def test_flux_pipeline_equivalence():
    """원본 flux_pipeline.py와 결과 비교"""
    # 동일한 시드로 두 파이프라인 실행
    # 결과 이미지 픽셀 레벨 비교 (허용 오차 내)
```

### 3. 배치 처리 테스트 (T005)
```bash
# uv run을 사용한 배치 테스트 실행
uv run tests/integration/test_batch_processing.py
```

```python
# tests/integration/test_batch_processing.py
def test_batch_prompts():
    """다중 프롬프트 배치 처리"""
    prompts = [
        "A beautiful landscape with mountains",
        "A futuristic city with flying cars",
        "A cat playing with a ball"
    ]

    # 배치 입력 생성
    inputs = create_batch_inputs(prompts)
    result = client.infer("bls", inputs)
    images = result.as_numpy("images")

    # 배치 출력 검증
    assert images.shape == (3, 3, 1024, 1024)

    # 각 이미지 개별 검증
    for i, prompt in enumerate(prompts):
        save_test_image(images[i], f"batch_test_{i}.png")

def test_batch_memory_efficiency():
    """배치 크기에 따른 메모리 사용량"""
    import GPUtil

    for batch_size in [1, 2, 4, 8]:
        initial_memory = GPUtil.getGPUs()[0].memoryUsed

        prompts = ["test prompt"] * batch_size
        result = run_batch_inference(prompts)

        peak_memory = GPUtil.getGPUs()[0].memoryUsed
        memory_per_image = (peak_memory - initial_memory) / batch_size

        assert memory_per_image < MAX_MEMORY_PER_IMAGE
```

### 4. 에러 처리 테스트
```python
def test_invalid_inputs():
    """잘못된 입력 처리"""
    # 빈 프롬프트
    # 너무 긴 프롬프트 (토큰 제한 초과)
    # 잘못된 이미지 크기
    # 비정상적인 inference_steps

def test_memory_limit():
    """메모리 부족 상황 처리"""
    # 과도한 배치 크기
    # 대용량 이미지 요청

def test_model_failure_recovery():
    """개별 모델 실패 시 복구"""
    # 특정 모델 서비스 중단 시뮬레이션
    # 적절한 에러 메시지 반환 확인
```

## 성능 벤치마킹
```python
def benchmark_throughput():
    """처리량 성능 측정"""
    import time

    start_time = time.time()
    num_requests = 100

    for i in range(num_requests):
        run_single_inference("benchmark prompt")

    total_time = time.time() - start_time
    throughput = num_requests / total_time

    assert throughput > MIN_THROUGHPUT_THRESHOLD

def benchmark_latency():
    """지연시간 측정"""
    latencies = []

    for _ in range(50):
        start = time.time()
        run_single_inference("latency test")
        latency = time.time() - start
        latencies.append(latency)

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    assert avg_latency < MAX_AVG_LATENCY
    assert p95_latency < MAX_P95_LATENCY
```

## 검증 기준
- flux_pipeline.py와 픽셀 레벨 유사성 (>95%)
- 배치 처리 정확성 (개별 결과와 동일)
- 메모리 효율성 (선형 증가)
- 에러 처리 적절성 (명확한 메시지)
- 성능 목표 달성 (처리량, 지연시간)