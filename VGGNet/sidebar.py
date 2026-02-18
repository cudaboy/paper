import streamlit as st
import model_trainer

def show():
    """Shows the sidebar and returns the configured parameters."""
    # 사이드바 제목 설정
    st.sidebar.title("VGG Model Training")
    st.sidebar.markdown("---")

    # 1. 데이터 업로드 섹션
    st.sidebar.header("Step 1: 데이터셋 업로드")
    uploaded_file = st.sidebar.file_uploader(
        "데이터셋 파일을 업로드하세요. (zip, tar.gz, pkl)", 
        type=['zip', 'tar.gz', 'pkl', 'pickle']
    )
    # 데이터셋 구조 예시 정보 제공
    st.sidebar.info(
        """**zip 파일 구조 예시:**
```
my_dataset.zip
└── my_dataset/
    ├── train/
    │   ├── class_A/
    │   └── class_B/
    └── val/
        ├── class_A/
        └── class_B/
```"""
    )

    st.sidebar.markdown("---")

    # 2. 모델 설정 섹션
    st.sidebar.header("Step 2: 모델 설정")
    # VGG 모델 버전 선택
    model_name = st.sidebar.selectbox("VGG 모델 선택", list(model_trainer.cfgs.keys()))
    # 배치 정규화 사용 여부 선택
    use_batch_norm = st.sidebar.checkbox("배치 정규화(Batch Normalization) 사용", value=True)

    st.sidebar.markdown("---")

    # 3. 학습 설정 섹션
    st.sidebar.header("Step 3: 학습 설정")
    
    # 데이터 처리 관련 설정
    st.sidebar.subheader("데이터 처리 설정")
    # 데이터 정규화 방식 선택
    normalize_option_label = st.sidebar.radio(
        "데이터 정규화(Normalization) 방식",
        ['ImageNet 통계 사용', '업로드한 데이터셋 통계 사용'],
        help="ImageNet 통계는 사전 학습된 모델에 이상적이며, 데이터셋 통계는 데이터셋 고유의 분포를 학습할 때 유용합니다."
    )
    normalize_option = 'imagenet' if normalize_option_label == 'ImageNet 통계 사용' else 'dataset'

    # 사용할 데이터 비율 조절
    subset_ratio = st.sidebar.slider("사용할 데이터 비율", min_value=0.1, max_value=1.0, value=1.0, step=0.1, help="학습 및 검증에 사용할 데이터의 비율을 조절합니다. 1.0은 전체 데이터를 의미합니다.")
    # 재현성을 위한 랜덤 시드 설정
    random_state = st.sidebar.number_input("Random Seed", value=42, min_value=0, help="데이터 샘플링 및 학습 과정의 재현성을 위한 시드 값입니다.")

    # 하이퍼파라미터 설정
    st.sidebar.subheader("하이퍼파라미터")
    epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=1000, value=10)
    batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=32)

    # Optimizer 관련 설정
    st.sidebar.subheader("Optimizer 설정")
    optimizer_name = st.sidebar.selectbox("Optimizer", ["SGD", "Adam", "Adagrad"])
    learning_rate = st.sidebar.number_input("Learning Rate", value=0.01, format="%.4f")
    weight_decay = st.sidebar.number_input("Weight Decay", value=0.0005, format="%.4f", help="L2 페널티. Adam의 경우 PyTorch 구현상의 이유로 실제로는 L2 디케이와 다르게 동작할 수 있습니다.")

    # SGD Optimizer 선택 시에만 Momentum 설정
    momentum = None
    if optimizer_name == "SGD":
        momentum = st.sidebar.number_input("Momentum", value=0.9, format="%.2f")

    # 학습 시작 버튼
    st.sidebar.markdown("---")
    start_button = st.sidebar.button("🚀 학습 시작!", disabled=(uploaded_file is None))

    # 설정된 파라미터들을 딕셔너리 형태로 반환
    return {
        "uploaded_file": uploaded_file,
        "model_name": model_name,
        "use_batch_norm": use_batch_norm,
        "normalize_option": normalize_option,
        "subset_ratio": subset_ratio,
        "random_state": random_state,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "start_button": start_button,
    }
