#!/usr/bin/env python3
"""
개별 모델 실행 테스트 스크립트

각 Triton 모델의 model.py 파일이 직접 실행 가능한지 검증합니다.
CLAUDE.md의 요구사항에 따라 model.py 직접 실행이 1순위 테스트 방법입니다.
"""

import sys
import os
import subprocess
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRITON_MODELS_DIR = PROJECT_ROOT / "triton_models"

def test_model_execution(model_name: str) -> bool:
    """개별 모델의 model.py 실행 테스트

    Args:
        model_name: 모델 이름 (bls, clip_encoder, t5_encoder, dit_transformer, vae_decoder)

    Returns:
        bool: 테스트 성공 여부
    """
    model_path = TRITON_MODELS_DIR / model_name / "1" / "model.py"

    if not model_path.exists():
        print(f"❌ {model_name}: model.py 파일이 존재하지 않음 ({model_path})")
        return False

    try:
        # DLPack 사용 여부에 따른 분기 처리 (헌장 요구사항)
        env = os.environ.copy()
        env["TEST_MODE"] = "true"  # 테스트 모드 활성화

        result = subprocess.run(
            [sys.executable, str(model_path)],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print(f"✅ {model_name}: 정상 실행 완료")
            if result.stdout:
                print(f"   출력: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {model_name}: 실행 실패 (exit code: {result.returncode})")
            if result.stderr:
                print(f"   에러: {result.stderr.strip()}")
            return False

    except subprocess.TimeoutExpired:
        print(f"❌ {model_name}: 실행 타임아웃 (30초)")
        return False
    except Exception as e:
        print(f"❌ {model_name}: 예외 발생 - {e}")
        return False

def main():
    """모든 개별 모델 테스트 실행"""
    print("🚀 개별 모델 실행 테스트 시작")
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"Triton 모델 디렉토리: {TRITON_MODELS_DIR}")
    print("-" * 50)

    # 테스트할 모델 목록 (contracts에서 확인한 5개 모델)
    models = [
        "clip_encoder",
        "t5_encoder",
        "dit_transformer",
        "vae_decoder",
        "bls"  # BLS는 다른 모델들에 의존하므로 마지막에 테스트
    ]

    results = {}

    for model in models:
        print(f"\n📋 {model} 테스트 중...")
        results[model] = test_model_execution(model)

    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)

    success_count = sum(results.values())
    total_count = len(results)

    for model, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{model:15} : {status}")

    print(f"\n총 {total_count}개 모델 중 {success_count}개 성공")

    if success_count == total_count:
        print("🎉 모든 개별 모델 테스트 통과!")
        return True
    else:
        print(f"⚠️  {total_count - success_count}개 모델에서 실패 발생")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)