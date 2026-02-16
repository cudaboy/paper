import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import data_handler
import model_trainer
import plot_utils
from utils import set_seed

def run(params):
    """Runs the main application logic."""
    st.title("Train Your Own VGG Model 🧠")
    st.markdown("---")

    if not params["start_button"] and not st.session_state.training_started:
        st.info("👈 사이드바에서 설정을 완료하고 '학습 시작' 버튼을 눌러주세요.")

    if params["start_button"]:
        st.session_state.training_started = True
        st.session_state.history = {}
        st.session_state.final_results = {}

        set_seed(params["random_state"])

        with st.spinner("데이터 처리 중... 잠시만 기다려주세요."):
            train_loader, val_loader, num_classes = data_handler.create_dataloaders(
                uploaded_file=params["uploaded_file"],
                upload_dir='uploads',
                batch_size=params["batch_size"],
                subset_ratio=params["subset_ratio"],
                random_state=params["random_state"],
                normalize_option=params["normalize_option"]
            )

        if train_loader is None:
            st.error("데이터 로더 생성에 실패했습니다. zip 파일 구조를 확인해주세요.")
            st.stop()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_trainer.build_vgg_model(
            params["model_name"], num_classes, params["use_batch_norm"]
        ).to(device)

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

        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3
        )

        st.header("📊 학습 과정")

        st.subheader("Model Summary")
        with st.expander("모델 구조 보기"):
            st.code(f"Device: {device}")
            st.code(
                f"Selected Model: {params['model_name'].upper()} {'with Batch Norm' if params['use_batch_norm'] else ''}"
            )
            st.code(f"Number of Classes: {num_classes}")
            model_summary = summary(
                model, input_size=(params["batch_size"], 3, 224, 224), verbose=0
            )
            st.text(str(model_summary))

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        epoch_progress_bar = st.progress(0)
        graph_container = st.empty()
        results_container = st.empty()

        for epoch in range(params["epochs"]):
            epoch_progress_bar.progress((epoch + 1) / params["epochs"], text=f"Epoch {epoch + 1}/{params['epochs']}")

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

            train_loss = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
            train_acc = train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
            val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
            val_acc = val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc

            scheduler.step(val_acc)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            st.session_state.history = history

            with graph_container:
                st.subheader("실시간 학습 현황")
                plot_utils.plot_history(history)

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
