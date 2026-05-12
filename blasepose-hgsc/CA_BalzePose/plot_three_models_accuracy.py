import json
import matplotlib.pyplot as plt

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'serif'


def load_accuracy(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["train_accuracy"]


def main():
    # 读取三个 JSON 文件的 train_accuracy
    # acc_openpose = load_accuracy("train_record_ultra_smooth.json")
    # acc_ca_blazepose = load_accuracy("train_record_improved.json")
    # acc_blazepose = load_accuracy("train_record_correct_regression.json")

    acc_openpose = load_accuracy("train_record_experiment1.json")
    acc_ca_blazepose = load_accuracy("train_record_experiment2.json")
    acc_blazepose = load_accuracy("train_record_experiment3.json")


    # epoch 从 1 开始
    epochs_openpose = range(1, len(acc_openpose) + 1)
    epochs_ca_blazepose = range(1, len(acc_ca_blazepose) + 1)
    epochs_blazepose = range(1, len(acc_blazepose) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # 画三条曲线（图例顺序：CA-BlazePose、BlazePose、OpenPose）
    line_ca = ax.plot(epochs_ca_blazepose, acc_ca_blazepose, "g-", linewidth=2, label="CA-BlazePose")
    line_blaze = ax.plot(epochs_blazepose, acc_blazepose, "r-", linewidth=2, label="BlazePose")
    line_open = ax.plot(epochs_openpose, acc_openpose, "b-", linewidth=2, label="OpenPose")

    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    handles = [line_ca[0], line_blaze[0], line_open[0]]
    labels = ["Proposed Method", "BlazePose$^{[19]}$", "OpenPose$^{[17]}$"]
    ax.legend(handles, labels, loc="upper left", fontsize=14)

    plt.tight_layout()
    output_path = "training_curves_three_models.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"三模型准确率曲线已生成并保存为 '{output_path}'")


if __name__ == "__main__":
    main()

