import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# =========================================
# 1. 定义 FonaDyn colormap 和 caxis
# =========================================
def fona_dyn_colors(metric):
    """Return colors array (N×3), minVal, maxVal following MATLAB FonaDynColors.m"""
    if metric.lower() == "total":
        intensity = np.linspace(0.95, 0.25, 71)
        colors = np.stack([intensity, intensity, intensity], axis=1)
        return colors, 0, 4
    elif metric.lower() == "clarity":
        intensity = np.linspace(0.49, 1.0, 52)
        colors = np.stack([np.zeros_like(intensity), intensity, np.zeros_like(intensity)], axis=1)
        return colors, 0.96, 1
    elif metric.lower() == "crest":
        hues = np.linspace(0.33, 0, 61)
        hsvs = np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=1)
        colors = hsv_to_rgb(hsvs)
        return colors, 1.414, 4
    elif metric.lower() == "specbal":
        hues = np.linspace(0.33, 0, 61)
        hsvs = np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=1)
        colors = hsv_to_rgb(hsvs)
        return colors, -40, -10  # 修正范围：-40到-10
    elif metric.lower() == "cpp":
        hues = np.linspace(0.666, 0, 34)
        hsvs = np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=1)
        colors = hsv_to_rgb(hsvs)
        return colors, 0, 20
    elif metric.lower() == "entropy":
        reds = np.linspace(0.9, 165/255, 31)
        greblus = np.linspace(1.0, 42/255, 31)
        colors = np.stack([reds, greblus, greblus], axis=1)
        return colors, 0, 20  # 修正范围：0到20
    elif metric.lower() == "deggmax":
        hues = np.linspace(0.33, 0, 67)
        hsvs = np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=1)
        colors = hsv_to_rgb(hsvs)
        return colors, 0, 1  # log10后的范围：0到1
    elif metric.lower() == "qcontact":
        hues = np.linspace(0.83, 0, 74)
        hsvs = np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=1)
        colors = hsv_to_rgb(hsvs)
        return colors, 0.1, 0.6
    elif metric.lower() == "icontact":
        hues = np.linspace(0.67, 0, 68)
        hsvs = np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=1)
        colors = hsv_to_rgb(hsvs)
        return colors, 0, 0.6
    else:
        # 默认 viridis
        return plt.cm.viridis(np.linspace(0,1,64)), None, None


# =========================================
# 2. 读取 VRP CSV
# =========================================
def load_vrp(filepath):
    df = pd.read_csv(filepath, sep=';', engine='python')
    df.columns = [c.strip() for c in df.columns]
    return df


# =========================================
# 3. 单文件绘图
# =========================================
def plot_vrp(df, filename, output_dir):
    metrics = ["Total", "Clarity", "Crest", "SpecBal", "CPP", "Entropy", "dEGGmax", "Qcontact", "Icontact"]
    xmin, xmax = 32, 91
    ymin, ymax = 30, 120

    # 调整图形尺寸以实现1.5:1的比例
    # 3x3网格，每个子图1.5:1，加上间距，整体应该是横向拉长的
    fig, axes = plt.subplots(3,3, figsize=(24,12))  # 24/12 = 2.0，为横向拉长留出更多空间
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    fig.suptitle(os.path.basename(filename), fontsize=16, weight='bold')

    for ax, metric in zip(axes.flat, metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        # 获取 colormap 和 caxis
        colors, cmin, cmax = fona_dyn_colors(metric)
        nColors = len(colors)

        # 构建像素矩阵 - 按照MATLAB的1.5:1比例 (150x100)
        allPixels = np.full((150, 100), np.nan)  # 150行(dB), 100列(MIDI)

        for _, row in df.iterrows():
            x, y = int(row["MIDI"]), int(row["dB"])
            if x < xmin or x > xmax or y < ymin or y > ymax:
                continue
            z = row[metric]
            
            # log10 转换 - 按照MATLAB逻辑
            if metric in ["Total", "dEGGmax"]:
                if z <= 0:
                    z = 0.01
                z = np.log10(z)
            
            # 直接使用y和x作为索引，就像MATLAB: allPixels(y, x) = z
            # 但需要确保索引在有效范围内
            if 0 <= y < 150 and 0 <= x < 100:
                allPixels[y, x] = z

        # 创建自定义colormap
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        # 绘制 - 使用imshow样式，设置1.5:1的比例尺
        rows, cols = allPixels.shape
        aspect_ratio = (xmax - xmin) / (ymax - ymin) / 1.5  # 1.5:1 横向拉长
        im = ax.imshow(allPixels, origin='lower', cmap=cmap, vmin=cmin, vmax=cmax,
                       extent=[0.5, 100.5, 0.5, 150.5],  # 匹配矩阵大小 100x150
                       aspect=aspect_ratio)
        ax.set_title(metric, fontsize=12, weight='bold')
        ax.set_xlabel("MIDI")
        ax.set_ylabel("dB")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        
        # colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)

    # 输出
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0]+".png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================
# 4. 批量处理文件夹
# =========================================
def process_folder(root="."):
    result_dir = os.path.join(root, "result")
    os.makedirs(result_dir, exist_ok=True)

    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".csv"):
                filepath = os.path.join(dirpath, f)
                try:
                    df = load_vrp(filepath)
                    plot_vrp(df, filepath, result_dir)
                except Exception as e:
                    print(f"Skipped {filepath}: {e}")


# =========================================
# 5. 主入口
# =========================================
if __name__ == "__main__":
    print("FonaDyn VRP batch plotter started...")
    
    # 指定要处理的文件夹路径
    target_folder = r"H:\Python Academics\Voicemapping Folk singing\Results\singing"
    
    # 检查文件夹是否存在
    if not os.path.exists(target_folder):
        print(f"Error: Folder not found - {target_folder}")
        print("Please check if the path is correct")
    else:
        print(f"Processing folder: {target_folder}")
        process_folder(target_folder)
        print("All done! Check the 'result' folder for output PNG files.")
