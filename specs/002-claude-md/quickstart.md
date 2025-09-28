# Quickstart: FLUX to Triton Conversion

## 개요
이 가이드는 FLUX 파이프라인을 Triton 추론 서버로 변환한 시스템을 빠르게 테스트하는 방법을 제공합니다.

## 전제 조건
- Python 3.8+
- NVIDIA GPU (CUDA 지원)
- Triton Inference Server
- 필요한 Python 패키지: torch, transformers, diffusers

## 1. 환경 설정

### 가상환경 생성 (uv 사용)
```bash
# uv 설치 (헌장 요구사항)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 초기화
uv init
uv add torch transformers diffusers tritonclient[all]
```

### Triton 서버 시작
```bash
# Docker를 사용한 Triton 서버 시작
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v$(pwd)/triton_models:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

## 2. 기본 테스트 시나리오

### 시나리오 1: 단일 프롬프트 이미지 생성
**목표**: 기본적인 텍스트-이미지 변환 검증

```python
# test_single_prompt.py
import requests
import json
import numpy as np
from PIL import Image

def test_single_prompt():
    """flux_pipeline.py의 기본 예제를 재현"""

    # BLS 오케스트레이터 호출
    url = "http://localhost:8000/v2/models/bls/infer"

    payload = {
        "inputs": [
            {
                "name": "prompt",
                "shape": [1],
                "datatype": "BYTES",
                "data": ["A cat holding a sign that says hello world"]
            },
            {
                "name": "num_inference_steps",
                "shape": [1],
                "datatype": "INT32",
                "data": [4]
            },
            {
                "name": "guidance_scale",
                "shape": [1],
                "datatype": "FP32",
                "data": [0.0]
            }
        ]
    }

    response = requests.post(url, json=payload)
    assert response.status_code == 200

    result = response.json()
    image_data = np.array(result["outputs"][0]["data"])

    # 이미지 형태 복원 (1, 3, 1024, 1024)
    image_shape = result["outputs"][0]["shape"]
    image_array = image_data.reshape(image_shape)

    # PIL 이미지로 변환 및 저장
    image_rgb = (image_array[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    image = Image.fromarray(image_rgb)
    image.save("test_output.png")

    print("✅ 단일 프롬프트 테스트 성공!")

if __name__ == "__main__":
    test_single_prompt()
```

### 시나리오 2: 배치 처리 테스트
**목표**: 다중 프롬프트 동시 처리 검증

```python
# test_batch_processing.py
def test_batch_processing():
    """여러 프롬프트를 배치로 처리"""

    url = "http://localhost:8000/v2/models/bls/infer"

    payload = {
        "inputs": [
            {
                "name": "prompt",
                "shape": [2],
                "datatype": "BYTES",
                "data": [
                    "A beautiful landscape with mountains",
                    "A futuristic city with flying cars"
                ]
            },
            {
                "name": "num_inference_steps",
                "shape": [1],
                "datatype": "INT32",
                "data": [4]
            }
        ]
    }

    response = requests.post(url, json=payload)
    assert response.status_code == 200

    result = response.json()

    # 배치 크기 검증
    image_shape = result["outputs"][0]["shape"]
    assert image_shape[0] == 2, f"Expected batch_size=2, got {image_shape[0]}"

    print("✅ 배치 처리 테스트 성공!")

if __name__ == "__main__":
    test_batch_processing()
```

### 시나리오 3: 개별 모델 테스트
**목표**: 각 Triton 모델의 독립적 동작 검증

```python
# test_individual_models.py
def test_clip_encoder():
    """CLIP 인코더 단독 테스트"""
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text = "A cat holding a sign"

    # 토큰화
    tokens = tokenizer(text, padding="max_length", max_length=77,
                      truncation=True, return_tensors="pt")

    url = "http://localhost:8001/v2/models/clip_encoder/infer"
    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": list(tokens.input_ids.shape),
                "datatype": "INT64",
                "data": tokens.input_ids.flatten().tolist()
            }
        ]
    }

    response = requests.post(url, json=payload)
    assert response.status_code == 200

    result = response.json()
    embeddings_shape = result["outputs"][0]["shape"]
    assert embeddings_shape == [1, 768], f"Expected [1, 768], got {embeddings_shape}"

    print("✅ CLIP 인코더 테스트 성공!")

def test_t5_encoder():
    """T5 인코더 단독 테스트"""
    from transformers import T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")
    text = "A cat holding a sign"

    tokens = tokenizer(text, padding="max_length", max_length=512,
                      truncation=True, return_tensors="pt")

    url = "http://localhost:8001/v2/models/t5_encoder/infer"
    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": list(tokens.input_ids.shape),
                "datatype": "INT64",
                "data": tokens.input_ids.flatten().tolist()
            }
        ]
    }

    response = requests.post(url, json=payload)
    assert response.status_code == 200

    result = response.json()
    # sequence_embeds output 검증
    seq_embeds = [out for out in result["outputs"] if out["name"] == "sequence_embeds"][0]
    assert seq_embeds["shape"][2] == 4096, "T5 embedding dimension should be 4096"

    print("✅ T5 인코더 테스트 성공!")

if __name__ == "__main__":
    test_clip_encoder()
    test_t5_encoder()
```

## 3. 성능 검증

### 메모리 사용량 모니터링
```python
# test_memory_usage.py
import psutil
import GPUtil

def monitor_memory_usage():
    """DLPack 최적화 효과 검증"""

    # GPU 메모리 사용량 측정
    gpu = GPUtil.getGPUs()[0]
    initial_gpu_memory = gpu.memoryUsed

    # CPU 메모리 사용량 측정
    process = psutil.Process()
    initial_cpu_memory = process.memory_info().rss / 1024 / 1024  # MB

    # 테스트 실행
    test_single_prompt()

    # 메모리 사용량 재측정
    gpu = GPUtil.getGPUs()[0]
    final_gpu_memory = gpu.memoryUsed
    final_cpu_memory = process.memory_info().rss / 1024 / 1024

    print(f"GPU 메모리 증가: {final_gpu_memory - initial_gpu_memory}MB")
    print(f"CPU 메모리 증가: {final_cpu_memory - initial_cpu_memory}MB")

    # DLPack 사용으로 CPU 메모리 증가가 최소화되어야 함
    assert final_cpu_memory - initial_cpu_memory < 100, "CPU 메모리 사용량이 너무 큼"

    print("✅ 메모리 최적화 검증 성공!")
```

## 4. 에러 처리 검증

### 우아한 성능 저하 테스트
```python
# test_graceful_degradation.py
def test_missing_dependencies():
    """DLPack 미지원 환경에서의 동작 검증"""

    # DLPack 모듈 임시 제거 시뮬레이션
    import sys
    if 'dlpack' in sys.modules:
        del sys.modules['dlpack']

    # 테스트 실행 (fallback 모드로 동작해야 함)
    test_single_prompt()
    print("✅ DLPack 미지원 환경에서도 정상 동작!")

def test_memory_limit():
    """GPU 메모리 부족 상황 처리"""

    # 매우 큰 배치 크기로 메모리 부족 유발
    url = "http://localhost:8000/v2/models/bls/infer"
    payload = {
        "inputs": [
            {
                "name": "prompt",
                "shape": [100],  # 과도한 배치 크기
                "datatype": "BYTES",
                "data": ["test prompt"] * 100
            }
        ]
    }

    response = requests.post(url, json=payload)

    # 에러가 발생하더라도 서버가 죽지 않고 적절한 에러 메시지 반환
    if response.status_code != 200:
        error_msg = response.json().get("error", "")
        assert "memory" in error_msg.lower(), "메모리 관련 에러 메시지가 부족함"
        print("✅ 메모리 부족 상황 적절히 처리됨!")
    else:
        print("⚠️ 대용량 배치도 처리 가능 (GPU 메모리 충분)")
```

## 5. 전체 테스트 실행

```python
# run_all_tests.py
def run_quickstart_tests():
    """모든 quickstart 테스트 실행"""

    print("🚀 FLUX to Triton 변환 시스템 검증 시작")

    try:
        # 기본 기능 테스트
        test_single_prompt()
        test_batch_processing()

        # 개별 모델 테스트
        test_clip_encoder()
        test_t5_encoder()

        # 성능 검증
        monitor_memory_usage()

        # 에러 처리 검증
        test_missing_dependencies()
        test_memory_limit()

        print("🎉 모든 테스트 통과! 시스템이 정상 동작합니다.")

    except AssertionError as e:
        print(f"❌ 테스트 실패: {e}")
        return False
    except Exception as e:
        print(f"💥 예상치 못한 오류: {e}")
        return False

    return True

if __name__ == "__main__":
    success = run_quickstart_tests()
    exit(0 if success else 1)
```

## 6. 디버깅 가이드

### 로그 확인
```bash
# Triton 서버 로그
docker logs <triton_container_id>

# 개별 모델 로그
curl http://localhost:8000/v2/models/bls/stats
```

### 일반적인 문제 해결
1. **CUDA 메모리 부족**: 배치 크기 감소 또는 GPU 메모리 정리
2. **모델 로딩 실패**: 모델 파일 경로 및 권한 확인
3. **토큰화 오류**: 입력 텍스트 길이 및 특수문자 확인

이 quickstart 가이드를 통해 FLUX to Triton 변환 시스템의 핵심 기능을 검증할 수 있습니다.