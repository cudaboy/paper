# VGG Model Trainer

이 프로젝트는 사용자가 직접 VGG 모델을 학습하고 실시간으로 결과를 확인할 수 있는 Streamlit 기반의 웹 애플리케이션입니다. VGG11, VGG13, VGG16, VGG19 모델을 선택하고, 다양한 하이퍼파라미터를 조정하여 자신만의 이미지 데이터셋에 대한 모델을 훈련할 수 있습니다.

## ✨ 주요 기능

- **GUI 기반 학습 환경**: 코드를 직접 수정할 필요 없이 웹 UI에서 모든 설정을 완료하고 학습을 시작할 수 있습니다.
- **다양한 VGG 모델 지원**: VGG11, VGG13, VGG16, VGG19 중에서 원하는 모델을 선택할 수 있습니다.
- **실시간 학습 모니터링**: 학습 및 검증 과정의 Loss와 Accuracy를 그래프와 표로 실시간 확인할 수 있습니다.
- **유연한 데이터셋 지원**:
    - `ImageFolder` 구조 (train/val 폴더)의 `.zip`, `.tar.gz` 파일
    - `CIFAR-10` 스타일의 데이터셋
    - 특정 형식의 `.pkl` 파일
- **상세한 하이퍼파라미터 설정**:
    - Optimizer (SGD, Adam, Adagrad)
    - Learning Rate, Weight Decay, Momentum
    - Epoch, Batch Size
    - 데이터 정규화 방식 (ImageNet 통계 / 데이터셋 자체 통계)
    - 배치 정규화(Batch Normalization) 사용 여부
- **재현성 보장**: Random Seed 고정을 통해 동일한 조건에서 동일한 결과를 얻을 수 있습니다.

## 🚀 실행 방법

1.  **저장소 복제 및 디렉토리 이동**:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository/VGGNet
    ```

2.  **필수 라이브러리 설치**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Streamlit 앱 실행**:
    ```bash
    streamlit run app.py
    ```

4.  **웹 브라우저 접속**:
    - 터미널에 나타나는 `Local URL` (예: `http://localhost:8501`)에 접속합니다.
    - 사이드바의 안내에 따라 데이터셋을 업로드하고 설정을 완료한 후 '학습 시작!' 버튼을 클릭합니다.

## 📂 프로젝트 구조

```
VGGNet/
├── app.py             # Streamlit 앱의 메인 실행 파일
├── main.py            # 데이터 처리, 모델 학습, 결과 표시 등 핵심 로직
├── sidebar.py         # Streamlit 사이드바 UI 및 파라미터 설정
├── model_trainer.py   # VGG 모델 아키텍처 및 학습/평가 함수 정의
├── data_handler.py    # 데이터셋 업로드, 전처리, DataLoader 생성
├── plot_utils.py      # 학습 결과(Loss, Accuracy) 시각화 유틸리티
├── utils.py           # 랜덤 시드 고정 등 기타 유틸리티 함수
├── requirements.txt   # 프로젝트 실행에 필요한 라이브러리 목록
└── uploads/           # 업로드된 데이터셋이 임시로 저장되는 폴더
```

## 💾 데이터셋 형식

이 애플리케이션은 다음과 같은 형식의 데이터셋을 지원합니다.

1.  **ImageFolder (`.zip`, `.tar.gz`)**:
    압축 파일 내에 `train`과 `val` 폴더가 있고, 각 폴더는 클래스별 하위 폴더를 포함해야 합니다.
    ```
    my_dataset.zip
    └── my_dataset/
        ├── train/
        │   ├── class_A/
        │   │   ├── image1.jpg
        │   │   └── image2.png
        │   └── class_B/
        │       ├── image3.jpg
        │       └── ...
        └── val/
            ├── class_A/
            │   ├── image10.jpg
            │   └── ...
            └── class_B/
                └── ...
    ```

2.  **Pickle (`.pkl`, `.pickle`)**:
    다음 키(key)를 포함하는 Python 딕셔너리 객체가 저장된 파일이어야 합니다.
    - `train_images`: (N, H, W, C) 형태의 NumPy 배열
    - `train_labels`: (N,) 형태의 NumPy 배열
    - `val_images`: (M, H, W, C) 형태의 NumPy 배열
    - `val_labels`: (M,) 형태의 NumPy 배열
    - `classes`: 클래스 이름 리스트 `['class_A', 'class_B', ...]`

## ⚙️ 설정 옵션 (사이드바)

- **Step 1: 데이터셋 업로드**: 학습에 사용할 데이터셋 파일을 업로드합니다.
- **Step 2: 모델 설정**:
    - **VGG 모델 선택**: `VGG11`, `VGG13`, `VGG16`, `VGG19` 중 선택합니다.
    - **배치 정규화 사용**: 모델의 각 합성곱 층 뒤에 배치 정규화를 추가할지 여부를 결정합니다.
- **Step 3: 학습 설정**:
    - **데이터 정규화 방식**:
        - `ImageNet 통계 사용`: ImageNet 데이터셋의 평균/표준편차로 정규화합니다.
        - `업로드한 데이터셋 통계 사용`: 업로드한 데이터셋의 평균/표준편차를 직접 계산하여 정규화합니다.
    - **사용할 데이터 비율**: 전체 데이터 중 학습/검증에 사용할 비율을 조절합니다 (예: 0.5 = 50%).
    - **Random Seed**: 데이터 분할 및 가중치 초기화 등에 사용될 랜덤 시드를 설정하여 재현성을 확보합니다.
    - **Epochs**: 전체 데이터셋을 몇 번 반복하여 학습할지 설정합니다.
    - **Batch Size**: 한 번의 반복(iteration)에서 사용할 데이터 샘플의 개수를 설정합니다.
    - **Optimizer**: `SGD`, `Adam`, `Adagrad` 중 경사 하강법 알고리즘을 선택합니다.
    - **Learning Rate**: 가중치를 업데이트하는 스텝의 크기를 조절합니다.
    - **Weight Decay**: 가중치가 너무 커지는 것을 방지하는 L2 페널티 항의 크기를 조절합니다.
    - **Momentum** (SGD 선택 시): 이전의 그래디언트 업데이트 방향을 얼마나 유지할지 결정합니다.
