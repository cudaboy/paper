import torch
import torch.nn as nn


# --- VGG 모델 구조 정의 ---
# VGG 모델의 각 버전에 대한 레이어 구성을 딕셔너리로 정의합니다.
# 숫자는 컨볼루션 레이어의 출력 채널 수를 의미하고, 'M'은 맥스 풀링(Max Pooling) 레이어를 의미합니다.
cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    """
    VGG 모델 클래스 정의.
    `features`: 컨볼루션 레이어들로 구성된 특성 추출기.
    `num_classes`: 최종 분류할 클래스의 수.
    `init_weights`: 가중치 초기화 여부.
    """
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features  # 컨볼루션 블록
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # 출력 크기를 (7, 7)로 고정
        # 분류기(classifier)는 완전 연결(fully-connected) 레이어들로 구성됩니다.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """모델의 순전파 과정을 정의합니다."""
        x = self.features(x)       # 특성 추출
        x = self.avgpool(x)        # 평균 풀링
        x = torch.flatten(x, 1)    # 1차원 벡터로 평탄화
        x = self.classifier(x)     # 분류
        return x

    def _initialize_weights(self):
        """모델의 가중치를 초기화합니다."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 초기화 (ReLU 활성화 함수에 적합)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    """
    VGG 설정(cfg)에 따라 컨볼루션 레이어 시퀀스를 생성합니다.
    `batch_norm`: 배치 정규화(Batch Normalization) 사용 여부.
    """
    layers = []
    in_channels = 3  # 입력 채널 수 (RGB 이미지)
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # 배치 정규화 사용 시 Conv -> BatchNorm -> ReLU 순서
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def build_vgg_model(model_name, num_classes, batch_norm=True):
    """
    지정된 VGG 모델을 빌드하여 반환합니다.
    `model_name`: 'VGG11', 'VGG16' 등 cfgs 딕셔너리의 키.
    `num_classes`: 분류할 클래스의 수.
    `batch_norm`: 배치 정규화 사용 여부.
    """
    if model_name not in cfgs:
        raise ValueError(f"지원하지 않는 모델입니다. 다음 중 선택하세요: {list(cfgs.keys())}")
    
    # 설정에 맞는 특성 추출기(feature extractor)를 생성합니다.
    model_features = make_layers(cfgs[model_name], batch_norm=batch_norm)
    # VGG 모델 객체를 생성합니다.
    model = VGG(model_features, num_classes=num_classes)
    return model

def train_one_epoch(model, loader, optimizer, criterion, device, progress_bar):
    """
    한 에포크(epoch) 동안 모델을 학습시킵니다.
    """
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 데이터로더로부터 미니배치를 받아와 학습을 진행합니다.
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()      # 그래디언트 초기화
        outputs = model(images)    # 순전파
        loss = criterion(outputs, labels) # 손실 계산
        loss.backward()            # 역전파
        optimizer.step()           # 가중치 업데이트
        
        # 통계 집계
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Streamlit 진행률 표시줄을 업데이트합니다.
        progress = (i + 1) / len(loader)
        progress_bar.progress(progress, text=f"Training... {int(progress*100)}%")

    # 에포크의 평균 손실과 정확도를 계산합니다.
    avg_loss = running_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

def evaluate(model, loader, criterion, device, progress_bar):
    """
    주어진 데이터로더를 사용하여 모델을 평가합니다.
    """
    model.eval()  # 모델을 평가 모드로 설정
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 그래디언트 계산 비활성화
        # 데이터로더로부터 미니배치를 받아와 평가를 진행합니다.
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 통계 집계
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Streamlit 진행률 표시줄을 업데이트합니다.
            progress = (i + 1) / len(loader)
            progress_bar.progress(progress, text=f"Validating... {int(progress*100)}%")

    # 평균 손실과 정확도를 계산합니다.
    avg_loss = running_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc