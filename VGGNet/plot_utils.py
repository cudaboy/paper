import matplotlib.pyplot as plt
import streamlit as st

def plot_history(history):
    """
    학습 기록(history)을 받아 Loss와 Accuracy 그래프를 생성하고 Streamlit에 표시합니다.
    
    Args:
        history (dict): {'train_loss': [...], 'val_loss': [...], 'train_acc': [...], 'val_acc': [...]}
                        형태의 딕셔너리
    """
    if not history:
        st.warning("학습 기록이 없어 그래프를 생성할 수 없습니다.")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # (1) Loss 그래프
    ax1.plot(epochs, history['train_loss'], 'b-', marker='o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', marker='o', label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # (2) Accuracy 그래프
    ax2.plot(epochs, history['train_acc'], 'b-', marker='o', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', marker='o', label='Validation Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # Streamlit 앱에 그래프 표시
    st.pyplot(fig)
