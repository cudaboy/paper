import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
from torchinfo import summary

# --- 모듈 임포트 ---
import data_handler
import model_trainer
import plot_utils

def set_seed(seed):
    """재현성을 위해 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 아래 주석 처리된 옵션은 재현성을 보장하지만, 학습 속도를 저하시킬 수 있습니다.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

# --- 페이지 설정 ---
st.set_page_config(page_title="VGG Model Trainer", layout="wide")

# --- 세션 상태 초기화 ---
# 학습 과정과 결과를 저장하여 페이지 리로드 시에도 유지
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'history' not in st.session_state:
    st.session_state.history = {}
if 'final_results' not in st.session_state:
    st.session_state.final_results = {}

# --- 사이드바 UI ---
st.sidebar.title("VGG Model Training")
st.sidebar.markdown("---")

# 1. 데이터 업로드
st.sidebar.header("Step 1: 데이터셋 업로드")
uploaded_file = st.sidebar.file_uploader(
    "데이터셋 파일을 업로드하세요. (zip, tar.gz, pkl)", 
    type=['zip', 'tar.gz', 'pkl', 'pickle']
)
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

# 2. 모델 및 파라미터 선택
st.sidebar.header("Step 2: 모델 및 학습 설정")
model_name = st.sidebar.selectbox("VGG 모델 선택", list(model_trainer.cfgs.keys()))
use_batch_norm = st.sidebar.checkbox("배치 정규화(Batch Normalization) 사용", value=True)

st.sidebar.subheader("데이터 처리 설정")
normalize_option_label = st.sidebar.radio(
    "데이터 정규화(Normalization) 방식",
    ['ImageNet 통계 사용', '업로드한 데이터셋 통계 사용'],
    help="ImageNet 통계는 사전 학습된 모델에 이상적이며, 데이터셋 통계는 데이터셋 고유의 분포를 학습할 때 유용합니다."
)
normalize_option = 'imagenet' if normalize_option_label == 'ImageNet 통계 사용' else 'dataset'

subset_ratio = st.sidebar.slider("사용할 데이터 비율", min_value=0.1, max_value=1.0, value=1.0, step=0.1, help="학습 및 검증에 사용할 데이터의 비율을 조절합니다. 1.0은 전체 데이터를 의미합니다.")
random_state = st.sidebar.number_input("Random Seed", value=42, min_value=0, help="데이터 샘플링 및 학습 과정의 재현성을 위한 시드 값입니다.")

# 하이퍼파라미터
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=1000, value=10)
batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=32)

st.sidebar.subheader("Optimizer 설정")
optimizer_name = st.sidebar.selectbox("Optimizer", ["SGD", "Adam", "Adagrad"])
learning_rate = st.sidebar.number_input("Learning Rate", value=0.01, format="%.4f")
weight_decay = st.sidebar.number_input("Weight Decay", value=0.0005, format="%.4f", help="L2 페널티. Adam의 경우 PyTorch 구현상의 이유로 실제로는 L2 디케이와 다르게 동작할 수 있습니다.")

momentum = None
if optimizer_name == "SGD":
    momentum = st.sidebar.number_input("Momentum", value=0.9, format="%.2f")

# 3. 학습 시작 버튼
st.sidebar.markdown("---")
start_button = st.sidebar.button("🚀 학습 시작!", disabled=(uploaded_file is None))

# --- 메인 페이지 UI ---
st.title("Train Your Own VGG Model 🧠")
st.markdown("---")

if not start_button and not st.session_state.training_started:
    st.info("👈 사이드바에서 설정을 완료하고 '학습 시작' 버튼을 눌러주세요.")

if start_button:
    st.session_state.training_started = True
    st.session_state.history = {}
    st.session_state.final_results = {}
    
    set_seed(random_state) # 재현성을 위해 시드 설정

    with st.spinner("데이터 처리 중... 잠시만 기다려주세요."):
        # 데이터 포맷에 맞춰 데이터로더 생성
        train_loader, val_loader, num_classes = data_handler.create_dataloaders(
            uploaded_file=uploaded_file,
            upload_dir='uploads',
            batch_size=batch_size,
            subset_ratio=subset_ratio,
            random_state=random_state,
            normalize_option=normalize_option
        )

    if train_loader is None:
        st.error("데이터 로더 생성에 실패했습니다. zip 파일 구조를 확인해주세요.")
        st.stop()
    
    # 모델 빌드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_trainer.build_vgg_model(model_name, num_classes, use_batch_norm).to(device)

    # Optimizer, Loss, Scheduler 정의
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else: # Adagrad
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # --- 학습 과정 표시 ---
    st.header("📊 학습 과정")
    
    # 모델 구조 정보
    st.subheader("Model Summary")
    with st.expander("모델 구조 보기"):
        st.code(f"Device: {device}")
        st.code(f"Selected Model: {model_name.upper()} {'with Batch Norm' if use_batch_norm else ''}")
        st.code(f"Number of Classes: {num_classes}")
        model_summary = summary(model, input_size=(batch_size, 3, 224, 224), verbose=0)
        st.text(str(model_summary))

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Epoch 진행률 표시
    epoch_progress_bar = st.progress(0)
    
    # 그래프를 실시간으로 업데이트하기 위한 컨테이너
    graph_container = st.empty()
    
    # 결과 요약을 위한 컨테이너
    results_container = st.empty()

    for epoch in range(epochs):
        epoch_progress_bar.progress((epoch + 1) / epochs, text=f"Epoch {epoch + 1}/{epochs}")

        # 학습
        train_progress_bar = st.progress(0)
        train_loss, train_acc = model_trainer.train_one_epoch(model, train_loader, optimizer, criterion, device, train_progress_bar)
        train_progress_bar.empty()
        
        # 평가
        val_progress_bar = st.progress(0)
        val_loss, val_acc = model_trainer.evaluate(model, val_loader, criterion, device, val_progress_bar)
        val_progress_bar.empty()

        # Tensor 값을 Python float/int로 변환
        train_loss = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        train_acc = train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
        val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
        val_acc = val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc

        scheduler.step(val_acc)

        # 결과 기록
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 세션 상태 업데이트
        st.session_state.history = history

        # 실시간 그래프 업데이트
        with graph_container:
            st.subheader("실시간 학습 현황")
            plot_utils.plot_history(history)
        
        # 실시간 결과 업데이트
        with results_container.container():
            st.subheader("Epoch 별 결과")
            df_results = {
                'Epoch': list(range(1, epoch + 2)),
                'Train Loss': [f"{l:.4f}" for l in history['train_loss']],
                'Train Acc': [f"{a:.2f}%" for a in history['train_acc']],
                'Val Loss': [f"{l:.4f}" for l in history['val_loss']],
                'Val Acc': [f"{a:.2f}%" for a in history['val_acc']],
            }
            st.dataframe(df_results, use_container_width=True)


    st.success("🎉 모든 학습이 완료되었습니다!")
    
    # 최종 결과 저장
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    st.session_state.final_results = {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_train_acc': history['train_acc'][-1],
        'final_train_loss': history['train_loss'][-1],
        'final_val_acc': history['val_acc'][-1],
        'final_val_loss': history['val_loss'][-1],
    }

# --- 학습 완료 후 결과 표시 ---
if st.session_state.training_started and st.session_state.final_results:
    st.header("🏁 최종 결과 요약")
    
    results = st.session_state.final_results
    st.metric(
        label=f"최고 검증 정확도 (Epoch {results['best_epoch']})",
        value=f"{results['best_val_acc']:.2f}%"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("최종 학습 결과")
        st.write(f"**Loss:** {results['final_train_loss']:.4f}")
        st.write(f"**Accuracy:** {results['final_train_acc']:.2f}%")
    with col2:
        st.subheader("최종 검증 결과")
        st.write(f"**Loss:** {results['final_val_loss']:.4f}")
        st.write(f"**Accuracy:** {results['final_val_acc']:.2f}%")

    st.header("📈 최종 학습 그래프")
    plot_utils.plot_history(st.session_state.history)
