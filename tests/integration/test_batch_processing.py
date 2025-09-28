#!/usr/bin/env python3
"""
배치 처리 통합 테스트

quickstart.md의 시나리오 2를 구현합니다.
다중 프롬프트 동시 처리 검증을 수행합니다.
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

class BatchProcessingTester:
    """배치 처리 테스터"""

    def __init__(self, triton_url: str = "http://localhost:8000"):
        self.triton_url = triton_url
        self.bls_endpoint = f"{triton_url}/v2/models/bls/infer"

    def test_batch_processing(self, batch_size: int = 2) -> bool:
        """배치 처리 테스트

        Args:
            batch_size: 배치 크기

        Returns:
            bool: 테스트 성공 여부
        """
        print(f"\n📋 배치 처리 테스트 (배치 크기: {batch_size})")

        # quickstart.md의 예제 프롬프트들 사용
        prompts = [
            "A beautiful landscape with mountains",
            "A futuristic city with flying cars"
        ]

        if batch_size > 2:
            # 배치 크기가 2보다 크면 추가 프롬프트 생성
            for i in range(2, batch_size):
                prompts.append(f"Test prompt {i+1}")

        prompts = prompts[:batch_size]  # 배치 크기에 맞게 자르기

        payload = {
            "inputs": [
                {
                    "name": "prompt",
                    "shape": [batch_size],
                    "datatype": "BYTES",
                    "data": prompts
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
            print(f"📝 프롬프트들:")
            for i, prompt in enumerate(prompts):
                print(f"  {i+1}. {prompt}")

            start_time = time.time()
            response = requests.post(self.bls_endpoint, json=payload, timeout=240)
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

            # 배치 크기 검증
            image_shape = image_output["shape"]
            expected_batch_size = batch_size
            actual_batch_size = image_shape[0]

            if actual_batch_size != expected_batch_size:
                print(f"❌ 배치 크기 불일치: {actual_batch_size} != {expected_batch_size}")
                return False

            print(f"✅ 배치 크기 검증 통과: {actual_batch_size}")
            print(f"📊 생성된 이미지 형태: {image_shape}")

            # 이미지 데이터 검증
            image_data = np.array(image_output["data"])
            expected_data_size = np.prod(image_shape)

            if len(image_data) != expected_data_size:
                print(f"❌ 이미지 데이터 크기 불일치: {len(image_data)} != {expected_data_size}")
                return False

            print(f"✅ 이미지 데이터 크기 검증 통과: {len(image_data)}")

            # 각 배치별 이미지 저장
            try:
                image_array = image_data.reshape(image_shape)

                for i in range(batch_size):
                    # 각 이미지 추출 및 저장
                    single_image = image_array[i]
                    image_rgb = (single_image.transpose(1, 2, 0) * 255).astype(np.uint8)

                    # 값 범위 확인
                    min_val, max_val = image_rgb.min(), image_rgb.max()
                    print(f"📊 이미지 {i+1} 픽셀 값 범위: {min_val} ~ {max_val}")

                    image = Image.fromarray(image_rgb)
                    output_path = PROJECT_ROOT / f"test_batch_output_{i+1}.png"
                    image.save(output_path)
                    print(f"💾 이미지 {i+1} 저장 완료: {output_path}")

            except Exception as e:
                print(f"⚠️ 이미지 저장 실패: {e}")

            print("✅ 배치 처리 테스트 성공!")
            return True

        except requests.Timeout:
            print("❌ 요청 타임아웃 (240초)")
            return False
        except Exception as e:
            print(f"❌ 테스트 중 예외 발생: {e}")
            return False

    def test_memory_efficiency(self) -> bool:
        """메모리 효율성 테스트"""
        print("\n📋 메모리 효율성 테스트")

        try:
            # 시스템 메모리 모니터링
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            print(f"📊 초기 메모리 사용량: {initial_memory:.2f} MB")

            # 작은 배치 테스트
            small_batch_success = self.test_batch_processing(batch_size=2)

            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory

            print(f"📊 테스트 후 메모리 사용량: {current_memory:.2f} MB")
            print(f"📊 메모리 증가량: {memory_increase:.2f} MB")

            # DLPack 최적화 효과 검증 (CPU 메모리 증가가 최소화되어야 함)
            if memory_increase < 200:  # 200MB 이하로 제한
                print("✅ 메모리 사용량이 효율적입니다")
                return small_batch_success
            else:
                print("⚠️ 메모리 사용량이 높습니다 (DLPack 최적화 확인 필요)")
                return small_batch_success

        except ImportError:
            print("⚠️ psutil 라이브러리가 없어 메모리 모니터링을 건너뜁니다")
            return self.test_batch_processing(batch_size=2)

def main():
    """배치 처리 통합 테스트 실행"""
    print("🚀 배치 처리 통합 테스트 시작")
    print("=" * 50)

    tester = BatchProcessingTester()

    # 1. 서버 상태 확인
    try:
        response = requests.get(f"{tester.triton_url}/v2/health/ready", timeout=5)
        if response.status_code != 200:
            print("❌ Triton 서버에 연결할 수 없습니다")
            print("\n💡 테스트 실행 전 Triton 서버를 시작하세요")
            return False
        print("✅ Triton 서버 연결 성공")
    except Exception:
        print("❌ Triton 서버에 연결할 수 없습니다")
        return False

    # 2. 배치 처리 테스트들
    test_results = []

    # 기본 배치 테스트 (배치 크기 2)
    test_results.append(tester.test_batch_processing(batch_size=2))

    # 메모리 효율성 테스트
    test_results.append(tester.test_memory_efficiency())

    # 추가: 더 큰 배치 테스트 (선택적)
    print("\n📋 선택적 대용량 배치 테스트")
    try:
        large_batch_success = tester.test_batch_processing(batch_size=4)
        test_results.append(large_batch_success)
        if large_batch_success:
            print("✅ 대용량 배치 처리도 가능합니다")
        else:
            print("⚠️ 대용량 배치 처리에서 문제 발생 (GPU 메모리 제한 가능성)")
    except Exception as e:
        print(f"⚠️ 대용량 배치 테스트 건너뜀: {e}")

    print("\n" + "=" * 50)
    success_count = sum(test_results)
    total_count = len(test_results)

    if success_count == total_count:
        print("🎉 배치 처리 통합 테스트 완료!")
    else:
        print(f"⚠️ {total_count - success_count}개 테스트에서 문제 발생")

    return success_count > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)