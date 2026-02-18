import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

# --- 모듈 임포트 ---
# 필요한 사용자 정의 모듈을 임포트합니다.
import data_handler  # 데이터 로딩 및 전처리
import model_trainer # 모델 빌드, 학습, 평가
import plot_utils    # 결과 시각화
from utils import set_seed # 재현성을 위한 시드 설정

def run(params):
    """
    애플리케이션의 메인 로직을 실행하는 함수입니다.
    사이드바에서 받은 `params`를 기반으로 데이터 로딩, 모델 학습, 결과 표시를 수행합니다.
    """
    st.title("Train Your Own VGG Model 🧠")
    st.markdown("---")

    # --- 초기 화면 ---
    # 학습이 시작되기 전 사용자에게 안내 메시지를 표시합니다.
    if not params["start_button"] and not st.session_state.training_started:
        st.info("👈 사이드바에서 설정을 완료하고 '학습 시작' 버튼을 눌러주세요.")

    # --- 학습 시작 ---
    # '학습 시작' 버튼이 클릭되면 전체 학습 파이프라인을 실행합니다.
    if params["start_button"]:
        # 세션 상태를 초기화하여 새로운 학습을 준비합니다.
        st.session_state.training_started = True
        st.session_state.history = {}
        st.session_state.final_results = {}

        # 재현성을 위해 랜덤 시드를 설정합니다.
        set_seed(params["random_state"])

        # --- 데이터 로딩 ---
        # 스피너(spinner)를 표시하여 데이터 처리 중임을 알립니다.
        with st.spinner("데이터 처리 중... 잠시만 기다려주세요."):
            train_loader, val_loader, num_classes = data_handler.create_dataloaders(
                uploaded_file=params["uploaded_file"],
                upload_dir='uploads',
                batch_size=params["batch_size"],
                subset_ratio=params["subset_ratio"],
                random_state=params["random_state"],
                normalize_option=params["normalize_option"]
            )

        # 데이터 로더 생성에 실패하면 오류 메시지를 표시하고 중단합니다.
        if train_loader is None:
            st.error("데이터 로더 생성에 실패했습니다. zip 파일 구조를 확인해주세요.")
            st.stop()

        # --- 모델 및 최적화 설정 ---
        # 사용 가능한 경우 CUDA 장치를 사용하고, 그렇지 않으면 CPU를 사용합니다.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 선택된 VGG 모델을 빌드하고 장치로 이동합니다.
        model = model_trainer.build_vgg_model(
            params["model_name"], num_classes, params["use_batch_norm"]
        ).to(device)

        # 선택된 옵티마이저를 설정합니다.
        if params["optimizer_name"] == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=params["learning_rate"],
                momentum=params["momentum"],
                weight_decay=params["weight_decay"],
            )
        elif params["optimizer_name"] == "Adam":
            optimizer = optim.Adam(
                model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
            )
        else:  # Adagrad
            optimizer = optim.Adagrad(
                model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
            )

        # 손실 함수(CrossEntropyLoss)와 스케줄러(ReduceLROnPlateau)를 설정합니다.
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3 # 검증 정확도가 3 epoch 동안 개선되지 않으면 학습률을 0.1배로 줄입니다.
        )

        st.header("📊 학습 과정")

        # --- 모델 요약 정보 표시 ---
        st.subheader("Model Summary")
        with st.expander("모델 구조 보기"):
            st.code(f"Device: {device}")
            st.code(
                f"Selected Model: {params['model_name'].upper()} {'with Batch Norm' if params['use_batch_norm'] else ''}"
            )
            st.code(f"Number of Classes: {num_classes}")
            # torchinfo.summary를 사용하여 모델의 상세 정보를 출력합니다.
            model_summary = summary(
                model, input_size=(params["batch_size"], 3, 224, 224), verbose=0
            )
            st.text(str(model_summary))

        # --- 학습 루프 ---
        # 학습 및 검증 결과를 저장할 딕셔너리를 초기화합니다.
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        # Streamlit UI 요소들을 초기화합니다.
        epoch_progress_bar = st.progress(0)
        graph_container = st.empty()
        results_container = st.empty()

        for epoch in range(params["epochs"]):
            # 현재 에포크 진행 상황을 표시합니다.
            epoch_progress_bar.progress((epoch + 1) / params["epochs"], text=f"Epoch {epoch + 1}/{params['epochs']}")

            # --- 1 에포크 학습 및 평가 ---
            train_progress_bar = st.progress(0)
            train_loss, train_acc = model_trainer.train_one_epoch(
                model, train_loader, optimizer, criterion, device, train_progress_bar
            )
            train_progress_bar.empty()

            val_progress_bar = st.progress(0)
            val_loss, val_acc = model_trainer.evaluate(
                model, val_loader, criterion, device, val_progress_bar
            )
            val_progress_bar.empty()

            # 결과를 텐서에서 스칼라 값으로 변환합니다.
            train_loss = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
            train_acc = train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
            val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
            val_acc = val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc

            # 스케줄러를 업데이트합니다.
            scheduler.step(val_acc)

            # --- 결과 기록 및 시각화 ---
            # 현재 에포크의 결과를 history에 추가합니다.
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # 세션 상태에 history를 저장하여 페이지가 다시 로드되어도 유지되도록 합니다.
            st.session_state.history = history

            # 실시간 그래프를 업데이트합니다.
            with graph_container:
                st.subheader("실시간 학습 현황")
                plot_utils.plot_history(history)

            # 에포크별 결과 테이블을 업데이트합니다.
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

        # --- 최종 결과 저장 ---
        # 가장 높은 검증 정확도를 기록한 에포크를 찾습니다.
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1
        # 세션 상태에 최종 결과를 저장합니다.
        st.session_state.final_results = {
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'final_train_acc': history['train_acc'][-1],
            'final_train_loss': history['train_loss'][-1],
            'final_val_acc': history['val_acc'][-1],
            'final_val_loss': history['val_loss'][-1],
        }

    # --- 학습 완료 후 최종 결과 표시 ---
    # 세션 상태에 최종 결과가 저장되어 있는 경우, 요약 정보를 표시합니다.
    if st.session_state.training_started and st.session_state.final_results:
        st.header("🏁 최종 결과 요약")

        results = st.session_state.final_results
        # 가장 높은 검증 정확도를 메트릭으로 표시합니다.
        st.metric(
            label=f"최고 검증 정확도 (Epoch {results['best_epoch']})",
            value=f"{results['best_val_acc']:.2f}%"
        )

        # 최종 학습 및 검증 결과를 두 개의 열로 나누어 표시합니다.
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("최종 학습 결과")
            st.write(f"**Loss:** {results['final_train_loss']:.4f}")
            st.write(f"**Accuracy:** {results['final_train_acc']:.2f}%")
        with col2:
            st.subheader("최종 검증 결과")
            st.write(f"**Loss:** {results['final_val_loss']:.4f}")
            st.write(f"**Accuracy:** {results['final_val_acc']:.2f}%")

        # 최종 학습 그래프를 표시합니다.
        st.header("📈 최종 학습 그래프")
        plot_utils.plot_history(st.session_state.history)