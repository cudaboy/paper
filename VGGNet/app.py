import streamlit as st

# --- 모듈 임포트 ---
import main
import sidebar

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
main.run(params)
