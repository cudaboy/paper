import random
import numpy as np
import torch

def set_seed(seed):
    """
    재현성을 위해 다양한 라이브러리의 시드를 고정하는 함수입니다.
    
    Args:
        seed (int): 고정할 시드 값
    """
    # Python 내장 random 라이브러리 시드 고정
    random.seed(seed)
    
    # NumPy 라이브러리 시드 고정
    np.random.seed(seed)
    
    # PyTorch 라이브러리 시드 고정
    torch.manual_seed(seed)
    
    # GPU 사용 가능 시, CUDA 관련 시드 고정
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 아래 주석 처리된 옵션들은 완벽한 재현성을 보장하지만,
        # 학습 속도를 저하시킬 수 있으므로 필요시 활성화합니다.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
