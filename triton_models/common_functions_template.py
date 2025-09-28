# ===================================================================
# FLUX Triton Models 공통 함수 템플릿
# 각 model.py에 복사해서 사용할 공통 함수들
# Triton 호환성을 위해 외부 import 없이 자체 포함
# ===================================================================

import json
import sys
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ===================================================================
# 1. 의존성 체크 함수들
# ===================================================================

def check_triton_availability():
    """Triton Python Backend 가용성 체크 및 Mock 설정"""
    try:
        import triton_python_backend_utils as pb_utils
        return True, pb_utils
    except ImportError:
        logger.warning("Triton backend not available - running in test mode")

        # Mock pb_utils for testing
        class MockTensor:
            def __init__(self, name, data):
                self.name = name
                self.data = data

        class MockInferenceRequest:
            def __init__(self):
                pass

        class MockInferenceResponse:
            def __init__(self, output_tensors=None, error=None):
                self.output_tensors = output_tensors or []
                self.error_msg = error

        class MockTritonError:
            def __init__(self, message):
                self.message = message

        class MockPbUtils:
            Tensor = MockTensor
            InferenceRequest = MockInferenceRequest
            InferenceResponse = MockInferenceResponse
            TritonError = MockTritonError

            @staticmethod
            def get_input_tensor_by_name(request, name):
                return None

            @staticmethod
            def get_output_tensor_by_name(response, name):
                return None

            @staticmethod
            def async_execute(request):
                return None

            @staticmethod
            def sync_execute(request):
                return None

        return False, MockPbUtils()

def check_dlpack_availability():
    """DLPack 가용성 체크"""
    try:
        import torch.utils.dlpack
        from torch.utils.dlpack import to_dlpack, from_dlpack
        return True, to_dlpack, from_dlpack
    except ImportError:
        logger.warning("DLPack not available - falling back to standard tensor operations")
        return False, None, None

def check_model_dependencies(required_modules: List[str]) -> bool:
    """필수 의존성 체크"""
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        return False
    return True

# ===================================================================
# 2. 파라미터 추출 함수들
# ===================================================================

def extract_model_parameters(model_config: Dict) -> Dict[str, str]:
    """모델 config에서 모든 파라미터 추출"""
    params = {}
    for param in model_config.get('parameters', []):
        params[param['key']] = param['value']['string_value']
    return params

def get_model_parameter(model_config: Dict, key: str, default: str = "") -> str:
    """특정 파라미터 값 추출"""
    for param in model_config.get('parameters', []):
        if param['key'] == key:
            return param['value']['string_value']
    return default

# ===================================================================
# 3. 디바이스 및 환경 설정 함수들
# ===================================================================

def setup_device(prefer_cuda: bool = True):
    """최적 디바이스 설정"""
    try:
        import torch
        if prefer_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        return device
    except ImportError:
        logger.warning("PyTorch not available, device setup skipped")
        return None

def setup_logging(model_name: str):
    """모델별 로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[{model_name}] %(asctime)s - %(levelname)s - %(message)s'
    )

# ===================================================================
# 4. 에러 처리 함수들
# ===================================================================

def create_error_response(pb_utils, error_message: str, triton_available: bool):
    """표준화된 에러 응답 생성"""
    if triton_available:
        return pb_utils.InferenceResponse(
            output_tensors=[],
            error=pb_utils.TritonError(error_message)
        )
    else:
        # Test mode에서는 None 반환
        logger.error(f"Error (test mode): {error_message}")
        return None

def handle_model_error(pb_utils, triton_available: bool, error: Exception, context: str = ""):
    """모델 에러 처리 및 응답 생성"""
    error_msg = f"{context} 처리 중 오류: {str(error)}"
    logger.error(error_msg)
    return create_error_response(pb_utils, error_msg, triton_available)

# ===================================================================
# 5. 텐서 처리 함수들 (DLPack Fallback 포함)
# ===================================================================

def create_tensor_with_fallback(pb_utils, name: str, data, dlpack_available: bool, to_dlpack_func):
    """DLPack 우선, numpy fallback으로 텐서 생성"""
    if dlpack_available and hasattr(pb_utils.Tensor, 'from_dlpack'):
        try:
            import torch
            torch_tensor = torch.from_numpy(data) if not isinstance(data, torch.Tensor) else data
            dlpack = to_dlpack_func(torch_tensor)
            return pb_utils.Tensor.from_dlpack(name, dlpack)
        except Exception as e:
            logger.warning(f"DLPack tensor creation failed, falling back to numpy: {e}")

    # Fallback to standard tensor creation
    import numpy as np
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return pb_utils.Tensor(name, data)

def extract_tensor_with_fallback(response, name: str, dlpack_available: bool, from_dlpack_func):
    """DLPack 우선, numpy fallback으로 텐서 추출"""
    try:
        import triton_python_backend_utils as pb_utils
        output_tensor = pb_utils.get_output_tensor_by_name(response, name)
        if output_tensor is None:
            raise ValueError(f"Missing output tensor: {name}")

        if dlpack_available and hasattr(output_tensor, 'is_dlpack') and output_tensor.is_dlpack():
            # Use DLPack for efficient transfer
            dlpack = output_tensor.as_dlpack()
            torch_tensor = from_dlpack_func(dlpack)
            return torch_tensor.cpu().numpy()
        else:
            # Fallback to standard numpy conversion
            return output_tensor.as_numpy()
    except Exception as e:
        logger.warning(f"Tensor extraction failed: {e}")
        return None

# ===================================================================
# 6. 모델 초기화 헬퍼 함수들
# ===================================================================

def initialize_model_base(args: Dict, model_name: str, required_modules: List[str]):
    """모든 모델의 공통 초기화 로직"""
    # 로깅 설정
    setup_logging(model_name)
    logger.info(f"{model_name} 초기화 시작...")

    # 의존성 체크
    if not check_model_dependencies(required_modules):
        logger.error(f"{model_name} 의존성 체크 실패")
        sys.exit(1)

    # Triton & DLPack 가용성 체크
    triton_available, pb_utils = check_triton_availability()
    dlpack_available, to_dlpack, from_dlpack = check_dlpack_availability()

    # 모델 config 파싱
    model_config = json.loads(args['model_config'])
    params = extract_model_parameters(model_config)

    # 디바이스 설정
    device = setup_device()

    return {
        'triton_available': triton_available,
        'pb_utils': pb_utils,
        'dlpack_available': dlpack_available,
        'to_dlpack': to_dlpack,
        'from_dlpack': from_dlpack,
        'model_config': model_config,
        'params': params,
        'device': device
    }

# ===================================================================
# 7. 배치 처리 헬퍼 함수들
# ===================================================================

def process_requests_batch(requests, process_single_request_func, pb_utils, triton_available: bool):
    """표준화된 배치 요청 처리"""
    responses = []

    for request in requests:
        try:
            response = process_single_request_func(request)
            responses.append(response)
        except Exception as e:
            error_response = handle_model_error(pb_utils, triton_available, e, "Request")
            responses.append(error_response)

    return responses

# ===================================================================
# 사용 예시 (각 모델에서 이렇게 사용)
# ===================================================================

"""
# 각 model.py에서 사용하는 방법:

class TritonPythonModel:
    def initialize(self, args):
        # 공통 초기화
        init_result = initialize_model_base(
            args,
            "CLIP_Encoder",
            ["torch", "transformers"]
        )

        # 초기화 결과를 인스턴스 변수에 저장
        self.triton_available = init_result['triton_available']
        self.pb_utils = init_result['pb_utils']
        self.dlpack_available = init_result['dlpack_available']
        self.device = init_result['device']
        # ... etc

        # 모델별 특화 초기화
        self._load_model_specific()

    def execute(self, requests):
        return process_requests_batch(
            requests,
            self._process_single_request,
            self.pb_utils,
            self.triton_available
        )
"""