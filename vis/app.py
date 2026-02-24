"""배수지 수요예측 및 펌프 최적화 결과 시각화 메인 애플리케이션."""

import json
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from demo_loader import (
    get_reservoir_names_from_project,
    get_sample_optimization_from_project,
    get_sample_prediction_from_project,
    try_run_real_inference,
    try_run_real_optimization,
)

# matplotlib 한글 출력 설정
plt.rcParams["font.family"] = [
    "AppleGothic",
    "Apple SD Gothic Neo",
    "Malgun Gothic",
    "NanumGothic",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


def make_kpi_card(title, value, icon=""):
    """HTML KPI 카드를 생성합니다."""
    return f"""
    <div style="border: 1px solid #ccc; padding: 15px; background: white;">
        <h3 style="margin: 0; font-size: 0.9rem;">{icon} {title}</h3>
        <p style="margin: 10px 0 0 0; font-size: 1.6rem; font-weight: 700;">{value}</p>
    </div>
    """


def plot_prediction(data):
    """수요예측 데이터를 시각화 차트와 KPI 카드로 변환합니다."""
    pred_arr = np.array(data["prediction_data"], dtype=float)
    x_labels = [f"+{i*15}분" for i in range(len(pred_arr))]
    x_pos = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(x_pos, pred_arr, alpha=0.2)
    sns.lineplot(
        x=x_pos,
        y=pred_arr,
        linewidth=2.5,
        marker="o",
        markersize=6,
        ax=ax,
        label="예측 수요",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_title("배수지 수요 예측 (15분 간격)", fontsize=14, fontweight="bold")
    ax.set_ylabel("수요")
    ax.legend()
    fig.tight_layout()

    acc = data.get("accuracy", "N/A")
    kpi_steps = make_kpi_card("총 예측 구간", f"{len(pred_arr)} 구간", "⏳")
    kpi_time = make_kpi_card("예측 기준 시각", data.get("predict_from", "N/A"), "🕒")
    kpi_acc = make_kpi_card(
        "모델 정확도",
        f"{float(acc):.2%}" if str(acc).replace(".", "").isdigit() else acc,
        "🎯",
    )

    return fig, kpi_steps, kpi_time, kpi_acc


def load_sample_predict(suzy_id, predict_from):
    """샘플 수요예측 데이터를 로드하고 시각화합니다."""
    data = get_sample_prediction_from_project(suzy_id=int(suzy_id))
    data["predict_from"] = predict_from
    return json.dumps(data, indent=2, ensure_ascii=False), *plot_prediction(data)


def load_real_predict(suzy_id, predict_from):
    """실제 추론 엔진을 통해 예측을 수행하고 결과를 시각화합니다."""
    data = try_run_real_inference(int(suzy_id), predict_from)
    return json.dumps(data, indent=2, ensure_ascii=False), *plot_prediction(data)


def plot_optimization(records):
    """펌프 최적화 시뮬레이션 결과를 4분할 차트와 KPI로 변환합니다."""
    df = pd.DataFrame(records)
    x_labels = [pd.Timestamp(t).strftime("%H:%M") for t in df["timestamp"]]
    x_pos = np.arange(len(x_labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    (ax1, ax2), (ax3, ax4) = axes

    # 펌프 가동 대수 (계단형)
    ax1.step(x_pos, df["active_pumps"], where="post", linewidth=2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")
    ax1.set_title("펌프 가동 대수")
    ax1.set_ylabel("대수")

    # 배수지별 수위
    resv_names = get_reservoir_names_from_project()
    for rid in records[0]["sim_levels"].keys():
        y_vals = [r["sim_levels"][rid] for r in records]
        name = resv_names.get(int(rid), f"ID {rid}")
        sns.lineplot(x=x_pos, y=y_vals, label=name, linewidth=1.5, ax=ax2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2.set_title("배수지별 수위")
    ax2.set_ylabel("수위")
    ax2.legend()

    # 시뮬레이션 비용
    sns.lineplot(x=x_pos, y=df["sim_cost"], linewidth=2, ax=ax3)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, rotation=45, ha="right")
    ax3.set_title("시뮬레이션 비용")
    ax3.set_ylabel("비용")

    # Spill량
    sns.barplot(x=x_pos, y=df["spill_m3_per_min"], ax=ax4)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(x_labels, rotation=45, ha="right")
    ax4.set_title("Spill량")
    ax4.set_ylabel("Spill (m³/min)")

    fig.tight_layout()

    return (
        fig,
        make_kpi_card("총 비용", f"₩ {df['sim_cost'].sum():,.0f}", "💰"),
        make_kpi_card("총 Spill", f"{df['spill_m3_per_min'].sum():,.4f}", "🌊"),
        make_kpi_card("구간", f"{len(df)} 구간", "⏱️"),
    )


def load_sample_optimize(_):
    """샘플 최적화 데이터를 로드하고 시각화합니다."""
    data = get_sample_optimization_from_project()
    return json.dumps(data, indent=2, ensure_ascii=False), *plot_optimization(data)


def load_real_optimize(start_time):
    """실제 최적화 엔진을 호출하여 결과를 시각화합니다."""
    data = try_run_real_optimization(start_time)
    return json.dumps(data, indent=2, ensure_ascii=False), *plot_optimization(data)


def build_ui():
    """Gradio 인터페이스를 구성하고 실행합니다."""
    with gr.Blocks(title="배수지 관제 대시보드") as demo:
        gr.Markdown("# 배수지 스마트 운영 관제")
        with gr.Tabs():
            with gr.TabItem(" 수요 예측"):
                with gr.Row():
                    with gr.Column(scale=1):
                        resv_dropdown = gr.Dropdown(
                            choices=[
                                (f"{n} ({i})", i)
                                for i, n in get_reservoir_names_from_project().items()
                            ],
                            value=next(iter(get_reservoir_names_from_project().keys())),
                            label="배수지 선택",
                        )
                        time_input = gr.Textbox(
                            value="2024-01-15 14:00:00", label="기준 시간"
                        )
                        with gr.Row():
                            btn_sample = gr.Button("샘플 생성")
                            btn_real = gr.Button("실제 예측", variant="primary")
                        json_out = gr.Textbox(label="결과 JSON", lines=15)
                    with gr.Column(scale=3):
                        with gr.Row():
                            k1, k2, k3 = gr.HTML(), gr.HTML(), gr.HTML()
                        plot_out = gr.Plot()
                btn_sample.click(
                    load_sample_predict,
                    [resv_dropdown, time_input],
                    [json_out, plot_out, k1, k2, k3],
                )
                btn_real.click(
                    load_real_predict,
                    [resv_dropdown, time_input],
                    [json_out, plot_out, k1, k2, k3],
                )

            with gr.TabItem("⚡ 펌프 최적화"):
                with gr.Row():
                    with gr.Column(scale=1):
                        opt_time = gr.Textbox(
                            value="2024-01-15 14:00:00", label="기준 시간"
                        )
                        with gr.Row():
                            btn_opt_sample = gr.Button("샘플 로드")
                            btn_opt_real = gr.Button("실제 최적화", variant="primary")
                        json_opt = gr.Textbox(label="결과 JSON", lines=15)
                    with gr.Column(scale=3):
                        with gr.Row():
                            ok1, ok2, ok3 = gr.HTML(), gr.HTML(), gr.HTML()
                        plot_opt = gr.Plot()
                btn_opt_sample.click(
                    load_sample_optimize,
                    [opt_time],
                    [json_opt, plot_opt, ok1, ok2, ok3],
                )
                btn_opt_real.click(
                    load_real_optimize, [opt_time], [json_opt, plot_opt, ok1, ok2, ok3]
                )
    return demo


if __name__ == "__main__":
    build_ui().launch()
