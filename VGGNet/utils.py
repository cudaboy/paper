import random
import numpy as np
import torch

def set_seed(seed):
    """재현성을 위해 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 재현성을 보장하지만, 학습 속도를 저하시킬 수 있는 옵션들
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
