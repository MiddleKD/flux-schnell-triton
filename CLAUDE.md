인터넷 검색이 필요할 경우 우선적으로 mcp-server-fetch 도구를 사용하세요.

# 개발 가이드

## 기술 스택
- **언어**: Python 3.8+
- **주요 의존성**: fvcore, ptflops, torch.profiler, onnxruntime, tensorrt
- **CLI 프레임워크**: Click + Rich
- **테스트**: pytest, pytest-cov
- **빌드**: Makefile 기반 명령어 통합
- **가상환경**: uv를 이용하고 .venv를 사용

## 아키텍처 원칙
- 어댑터 패턴으로 각 프레임워크별 도구 통합 (fvcore, ptflops, onnxruntime, trtexec)
- 계층화된 의존성: 핵심/선택적 패키지로 분리
- 우아한 성능 저하: 누락된 의존성 시 기능 비활성화

## 개발 가이드라인
- TDD 필수: 테스트 먼저 작성
- 파일당 500줄 이하 유지
- 단순성 우선: 복잡한 패턴 금지
- 실무 검증된 도구 활용: 자체 구현 최소화
