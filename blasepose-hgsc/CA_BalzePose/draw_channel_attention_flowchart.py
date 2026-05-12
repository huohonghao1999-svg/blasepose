# -*- coding: utf-8 -*-
"""绘制 Channel Attention 模块流程图，输出为 channel_attention_flowchart.png"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

def draw_flowchart():
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    box_w = 5.0
    box_h = 0.9
    cx = 5.0
    dy = 1.2

    def box(y, text, facecolor="lightblue"):
        r = FancyBboxPatch(
            (cx - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02",
            facecolor=facecolor,
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(r)
        ax.text(cx, y, text, ha="center", va="center", fontsize=11, wrap=True)

    def arrow(y_from, y_to):
        ax.annotate(
            "",
            xy=(cx, y_to - box_h / 2 - 0.05),
            xytext=(cx, y_from + box_h / 2 + 0.05),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        )

    # 自上而下：输入 → GAP → Dense1 → Dense2 → 逐通道乘 → 输出
    y_input = 12.5
    y_gap = 10.8
    y_d1 = 9.1
    y_d2 = 7.4
    y_mul = 5.7
    y_output = 4.0

    box(y_input, "输入特征\nInput  X  [B, H, W, C]", facecolor="#E8F4FD")
    arrow(y_input, y_gap)
    box(
        y_gap,
        "Global Average Pooling (GAP)\ngap = mean(X, axis=(1,2))  →  [B, 1, 1, C]",
        facecolor="#FFF4E6",
    )
    arrow(y_gap, y_d1)
    box(
        y_d1,
        "Dense1:  C → C/r  (ReLU, no bias)\n瓶颈维度",
        facecolor="#E8F5E9",
    )
    arrow(y_d1, y_d2)
    box(
        y_d2,
        "Dense2:  C/r → C  (Sigmoid, no bias)\n通道权重 [B, 1, 1, C]",
        facecolor="#E8F5E9",
    )
    arrow(y_d2, y_mul)
    box(
        y_mul,
        "逐通道相乘\nOutput = X ⊙ attention_weights",
        facecolor="#F3E5F5",
    )
    arrow(y_mul, y_output)
    box(y_output, "输出特征\nOutput  [B, H, W, C]", facecolor="#E8F4FD")

    plt.tight_layout()
    out_path = "channel_attention_flowchart.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"流程图已保存: {out_path}")


if __name__ == "__main__":
    draw_flowchart()
