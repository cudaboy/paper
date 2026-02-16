import random
import numpy as np
import torch
import streamlit as st

# --- 모듈 임포트 ---
import main
import sidebar

def set_seed(seed):
    """재현성을 위해 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

# --- 사이드바 및 메인 페이지 실행 ---
params = sidebar.show()
main.run(params, set_seed)
