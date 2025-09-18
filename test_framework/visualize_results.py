"""
可视化简化VRP结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_vrp_results(csv_file):
    """绘制VRP结果"""
    # 读取数据
    df = pd.read_csv(csv_file)
    
    print(f"=== VRP结果可视化 ===")
    print(f"数据点数量: {len(df)}")
    print(f"MIDI范围: {df['MIDI'].min():.1f} - {df['MIDI'].max():.1f}")
    print(f"SPL范围: {df['SPL'].min():.1f} - {df['SPL'].max():.1f}")
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 散点图
    ax1.scatter(df['MIDI'], df['SPL'], alpha=0.6, s=20)
    ax1.set_xlabel('MIDI')
    ax1.set_ylabel('SPL (dB)')
    ax1.set_title('VRP结果 - 散点图')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(30, 100)
    ax1.set_ylim(30, 120)
    
    # 热力图
    midi_bins = np.arange(30, 105, 5)
    spl_bins = np.arange(30, 125, 5)
    hist, _, _ = np.histogram2d(df['MIDI'], df['SPL'], 
                               bins=[midi_bins, spl_bins])
    im = ax2.imshow(hist.T, origin='lower', aspect='auto', 
                   extent=[30, 100, 30, 120], cmap='viridis')
    ax2.set_xlabel('MIDI')
    ax2.set_ylabel('SPL (dB)')
    ax2.set_title('VRP结果 - 热力图')
    plt.colorbar(im, ax=ax2, label='Count')
    
    plt.tight_layout()
    plt.savefig('vrp_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图片已保存为: vrp_result.png")

if __name__ == "__main__":
    plot_vrp_results("midi_spl_results.csv")
