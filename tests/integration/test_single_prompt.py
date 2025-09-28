#!/usr/bin/env python3
"""
단일 프롬프트 통합 테스트

quickstart.md의 시나리오 1을 구현합니다.
기본적인 텍스트-이미지 변환 검증을 수행합니다.
"""

import sys
import time
import numpy as np
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
    from PIL import Image
except ImportError:
    print("⚠️ requests 또는 PIL 라이브러리가 필요합니다")
    print("설치 명령: uv add requests pillow")
    sys.exit(1)

class FluxTritonTester:
    """FLUX to Triton 변환 시스템 테스터"""

    def __init__(self, triton_url: str = "http://localhost:8000"):
        self.triton_url = triton_url
        self.bls_endpoint = f"{triton_url}/v2/models/bls/infer"

    def test_server_health(self) -> bool:
        """Triton 서버 상태 확인"""
        try:
            response = requests.get(f"{self.triton_url}/v2/health/ready", timeout=5)
            if response.status_code == 200:
                print("✅ Triton 서버 연결 성공")
                return True
            else:
                print(f"❌ Triton 서버 응답 오류: {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"❌ Triton 서버 연결 실패: {e}")
            print("💡 Triton 서버가 실행 중인지 확인하세요")
            return False

    def test_single_prompt_generation(self) -> bool:
        """단일 프롬프트 이미지 생성 테스트"""
        print("\n📋 단일 프롬프트 이미지 생성 테스트")

        # flux_pipeline.py 예제와 동일한 프롬프트 사용
        prompt = "A cat holding a sign that says hello world"

        payload = {
            "inputs": [
                {
                    "name": "prompt",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [prompt]
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

        try:
            print(f"🔄 BLS 엔드포인트 호출: {self.bls_endpoint}")
            print(f"📝 프롬프트: {prompt}")

            start_time = time.time()
            response = requests.post(self.bls_endpoint, json=payload, timeout=120)
            elapsed_time = time.time() - start_time

            if response.status_code != 200:
                print(f"❌ API 호출 실패: {response.status_code}")
                print(f"응답 내용: {response.text}")
                return False

            result = response.json()
            print(f"✅ API 호출 성공 (소요시간: {elapsed_time:.2f}초)")

            # 응답 구조 검증
            if "outputs" not in result:
                print("❌ 응답에 'outputs' 필드가 없음")
                return False

            image_output = None
            for output in result["outputs"]:
                if output.get("name") == "images":
                    image_output = output
                    break

            if image_output is None:
                print("❌ 응답에 'images' 출력이 없음")
                return False

            # 이미지 형태 검증
            image_shape = image_output["shape"]
            expected_shape = [1, 3, 1024, 1024]  # [batch_size, channels, height, width]

            if image_shape != expected_shape:
                print(f"⚠️ 이미지 형태가 예상과 다름: {image_shape} (예상: {expected_shape})")

            print(f"📊 생성된 이미지 형태: {image_shape}")

            # 이미지 데이터 검증
            image_data = np.array(image_output["data"])
            if len(image_data) != np.prod(image_shape):
                print(f"❌ 이미지 데이터 크기 불일치: {len(image_data)} != {np.prod(image_shape)}")
                return False

            # 이미지 저장 시도
            try:
                image_array = image_data.reshape(image_shape)
                # 정규화된 값을 0-255로 변환
                image_rgb = (image_array[0].transpose(1, 2, 0) * 255).astype(np.uint8)

                # 값 범위 확인
                min_val, max_val = image_rgb.min(), image_rgb.max()
                print(f"📊 픽셀 값 범위: {min_val} ~ {max_val}")

                image = Image.fromarray(image_rgb)
                output_path = PROJECT_ROOT / "test_single_prompt_output.png"
                image.save(output_path)
                print(f"💾 이미지 저장 완료: {output_path}")

            except Exception as e:
                print(f"⚠️ 이미지 저장 실패: {e}")

            print("✅ 단일 프롬프트 테스트 성공!")
            return True

        except requests.Timeout:
            print("❌ 요청 타임아웃 (120초)")
            return False
        except Exception as e:
            print(f"❌ 테스트 중 예외 발생: {e}")
            return False

def main():
    """단일 프롬프트 통합 테스트 실행"""
    print("🚀 단일 프롬프트 통합 테스트 시작")
    print("=" * 50)

    tester = FluxTritonTester()

    # 1. 서버 상태 확인
    if not tester.test_server_health():
        print("\n💡 테스트 실행 전 Triton 서버를 시작하세요:")
        print("docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \\")
        print("  -v$(pwd)/triton_models:/models \\")
        print("  nvcr.io/nvidia/tritonserver:23.10-py3 \\")
        print("  tritonserver --model-repository=/models")
        return False

    # 2. 단일 프롬프트 테스트
    success = tester.test_single_prompt_generation()

    print("\n" + "=" * 50)
    if success:
        print("🎉 단일 프롬프트 통합 테스트 완료!")
    else:
        print("❌ 단일 프롬프트 통합 테스트 실패")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)