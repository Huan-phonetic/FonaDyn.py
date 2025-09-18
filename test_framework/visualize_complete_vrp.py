"""
可视化完整VRP结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_complete_vrp(csv_file):
    """绘制完整VRP结果"""
    # 读取数据
    df = pd.read_csv(csv_file)
    
    print(f"=== 完整VRP结果可视化 ===")
    print(f"数据点数量: {len(df)}")
    print(f"MIDI范围: {df['MIDI'].min()} - {df['MIDI'].max()}")
    print(f"SPL范围: {df['SPL'].min()} - {df['SPL'].max()}")
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. MIDI-SPL散点图
    scatter = axes[0, 0].scatter(df['MIDI'], df['SPL'], 
                                c=df['Clarity'], cmap='viridis', alpha=0.6, s=20)
    axes[0, 0].set_xlabel('MIDI')
    axes[0, 0].set_ylabel('SPL (dB)')
    axes[0, 0].set_title('VRP - Clarity')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(30, 100)
    axes[0, 0].set_ylim(30, 120)
    plt.colorbar(scatter, ax=axes[0, 0], label='Clarity')
    
    # 2. MIDI-SPL散点图 - CPP
    scatter2 = axes[0, 1].scatter(df['MIDI'], df['SPL'], 
                                 c=df['CPP'], cmap='plasma', alpha=0.6, s=20)
    axes[0, 1].set_xlabel('MIDI')
    axes[0, 1].set_ylabel('SPL (dB)')
    axes[0, 1].set_title('VRP - CPP')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(30, 100)
    axes[0, 1].set_ylim(30, 120)
    plt.colorbar(scatter2, ax=axes[0, 1], label='CPP')
    
    # 3. MIDI-SPL散点图 - Spectrum Balance
    scatter3 = axes[0, 2].scatter(df['MIDI'], df['SPL'], 
                                  c=df['SpectrumBalance'], cmap='coolwarm', alpha=0.6, s=20)
    axes[0, 2].set_xlabel('MIDI')
    axes[0, 2].set_ylabel('SPL (dB)')
    axes[0, 2].set_title('VRP - Spectrum Balance')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(30, 100)
    axes[0, 2].set_ylim(30, 120)
    plt.colorbar(scatter3, ax=axes[0, 2], label='Spectrum Balance')
    
    # 4. MIDI-SPL散点图 - Crest Factor
    scatter4 = axes[1, 0].scatter(df['MIDI'], df['SPL'], 
                                  c=df['CrestFactor'], cmap='hot', alpha=0.6, s=20)
    axes[1, 0].set_xlabel('MIDI')
    axes[1, 0].set_ylabel('SPL (dB)')
    axes[1, 0].set_title('VRP - Crest Factor')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(30, 100)
    axes[1, 0].set_ylim(30, 120)
    plt.colorbar(scatter4, ax=axes[1, 0], label='Crest Factor')
    
    # 5. MIDI-SPL散点图 - QCI
    scatter5 = axes[1, 1].scatter(df['MIDI'], df['SPL'], 
                                  c=df['QCI'], cmap='spring', alpha=0.6, s=20)
    axes[1, 1].set_xlabel('MIDI')
    axes[1, 1].set_ylabel('SPL (dB)')
    axes[1, 1].set_title('VRP - QCI')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(30, 100)
    axes[1, 1].set_ylim(30, 120)
    plt.colorbar(scatter5, ax=axes[1, 1], label='QCI')
    
    # 6. MIDI-SPL散点图 - dEGGmax
    scatter6 = axes[1, 2].scatter(df['MIDI'], df['SPL'], 
                                  c=df['dEGGmax'], cmap='autumn', alpha=0.6, s=20)
    axes[1, 2].set_xlabel('MIDI')
    axes[1, 2].set_ylabel('SPL (dB)')
    axes[1, 2].set_title('VRP - dEGGmax')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim(30, 100)
    axes[1, 2].set_ylim(30, 120)
    plt.colorbar(scatter6, ax=axes[1, 2], label='dEGGmax')
    
    plt.tight_layout()
    plt.savefig('complete_vrp_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图片已保存为: complete_vrp_result.png")

if __name__ == "__main__":
    plot_complete_vrp("complete_vrp_results.csv")
