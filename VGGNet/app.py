import streamlit as st

# --- 모듈 임포트 ---
# main.py와 sidebar.py 모듈을 임포트합니다.
import main
import sidebar

# --- 페이지 설정 ---
# Streamlit 페이지의 제목과 레이아웃을 설정합니다.
st.set_page_config(page_title="VGG Model Trainer", layout="wide")

# --- 세션 상태 초기화 ---
# 학습 과정과 결과를 저장하여 페이지 리로드 시에도 유지합니다.
# 'training_started'는 학습 시작 여부를 확인하는 변수입니다.
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
# 'history'는 학습 과정의 손실 및 정확도 기록을 저장하는 변수입니다.
if 'history' not in st.session_state:
    st.session_state.history = {}
# 'final_results'는 최종 학습 결과를 저장하는 변수입니다.
if 'final_results' not in st.session_state:
    st.session_state.final_results = {}

# --- 사이드바 및 메인 페이지 실행 ---
params = sidebar.show()
# 입력받은 파라미터로 메인 페이지를 실행합니다.
main.run(params)