import matplotlib.pyplot as plt
import streamlit as st

def plot_history(history):
    """
    학습 과정에서 기록된 'history' 딕셔너리를 사용하여
    손실(Loss)과 정확도(Accuracy) 그래프를 생성하고 Streamlit에 표시합니다.
    
    Args:
        history (dict): 다음과 같은 키를 포함하는 딕셔너리
                        - 'train_loss': 에포크별 학습 손실 리스트
                        - 'val_loss': 에포크별 검증 손실 리스트
                        - 'train_acc': 에포크별 학습 정확도 리스트
                        - 'val_acc': 에포크별 검증 정확도 리스트
    """
    # 학습 기록이 없는 경우, 경고 메시지를 표시하고 함수를 종료합니다.
    if not history or not history.get('train_loss'):
        st.warning("학습 기록이 없어 그래프를 생성할 수 없습니다.")
        return

    # 에포크 수를 계산합니다.
    epochs = range(1, len(history['train_loss']) + 1)

    # 1행 2열의 서브플롯을 생성합니다.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- (1) 손실(Loss) 그래프 ---
    # 학습 손실과 검증 손실을 그립니다.
    ax1.plot(epochs, history['train_loss'], 'b-', marker='o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', marker='o', label='Validation Loss')
    # 그래프의 제목, x축, y축 레이블을 설정합니다.
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    # 범례와 그리드를 표시합니다.
    ax1.legend()
    ax1.grid(True)

    # --- (2) 정확도(Accuracy) 그래프 ---
    # 학습 정확도와 검증 정확도를 그립니다.
    ax2.plot(epochs, history['train_acc'], 'b-', marker='o', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', marker='o', label='Validation Accuracy')
    # 그래프의 제목, x축, y축 레이블을 설정합니다.
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    # 범례와 그리드를 표시합니다.
    ax2.legend()
    ax2.grid(True)

    # 그래프 레이아웃을 조정합니다.
    plt.tight_layout()
    
    # 생성된 그래프를 Streamlit 앱에 표시합니다.
    st.pyplot(fig)