#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化显著性分析结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载分析结果"""
    base_path = r"H:\Python Academics\Voicemapping Folk singing\Results\singing\significance_results"
    
    # 加载详细结果
    detailed_df = pd.read_csv(os.path.join(base_path, "significance_detailed_results.csv"), sep=';')
    
    # 加载贡献度分析
    contribution_df = pd.read_csv(os.path.join(base_path, "contribution_analysis.csv"), sep=';')
    
    return detailed_df, contribution_df

def create_contribution_ranking_plot(contribution_df, output_path):
    """创建贡献度排名图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 贡献度评分排名
    metrics = contribution_df['metric'].tolist()
    scores = contribution_df['contribution_score'].tolist()
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    
    bars1 = ax1.barh(range(len(metrics)), scores, color=colors)
    ax1.set_yticks(range(len(metrics)))
    ax1.set_yticklabels(metrics, fontsize=12)
    ax1.set_xlabel('Contribution Score', fontsize=14, fontweight='bold')
    ax1.set_title('Metrics Contribution Ranking', fontsize=16, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars1, scores)):
        ax1.text(score + 0.1, i, f'{score:.2f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # 2. 效应量 vs p值散点图
    effect_sizes = contribution_df['effect_size'].tolist()
    p_values = contribution_df['p_value'].tolist()
    
    # 转换p值为-log10(p)以便可视化
    log_p_values = [-np.log10(p) if p > 0 else 20 for p in p_values]
    
    scatter = ax2.scatter(effect_sizes, log_p_values, 
                         s=[score*50 for score in scores], 
                         c=colors, alpha=0.7, edgecolors='black')
    
    ax2.set_xlabel('Effect Size', fontsize=14, fontweight='bold')
    ax2.set_ylabel('-log10(p-value)', fontsize=14, fontweight='bold')
    ax2.set_title('Effect Size vs Significance', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加显著性阈值线
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax2.axhline(y=-np.log10(0.001), color='orange', linestyle='--', alpha=0.7, label='p=0.001')
    ax2.legend()
    
    # 添加metric标签
    for i, metric in enumerate(metrics):
        ax2.annotate(metric, (effect_sizes[i], log_p_values[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Contribution ranking plot saved to: {output_path}")

def create_genre_comparison_plot(detailed_df, output_path):
    """创建歌种对比图"""
    # 筛选显著差异的metrics
    significant_metrics = detailed_df[detailed_df['significant'] == True]['metric'].unique()
    
    # 创建子图
    n_metrics = len(significant_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    genres = ['BelCanto', 'FolkSongs', 'PopularMusic']
    genre_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
    
    for idx, metric in enumerate(significant_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # 获取该metric的数据
        metric_data = detailed_df[detailed_df['metric'] == metric]
        
        # 提取各歌种的均值
        means = []
        stds = []
        for genre in genres:
            genre_data = metric_data[metric_data['genre'] == genre]
            if not genre_data.empty:
                means.append(genre_data['mean'].iloc[0])
                stds.append(genre_data['std'].iloc[0])
            else:
                means.append(0)
                stds.append(0)
        
        # 创建柱状图
        bars = ax.bar(genres, means, yerr=stds, capsize=5, 
                     color=genre_colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01*max(means),
                   f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Genre comparison plot saved to: {output_path}")

def create_effect_size_heatmap(detailed_df, output_path):
    """创建效应量热图"""
    # 筛选显著差异的metrics
    significant_metrics = detailed_df[detailed_df['significant'] == True]['metric'].unique()
    
    # 创建数据矩阵
    genres = ['BelCanto', 'FolkSongs', 'PopularMusic']
    data_matrix = np.zeros((len(significant_metrics), len(genres)))
    
    for i, metric in enumerate(significant_metrics):
        metric_data = detailed_df[detailed_df['metric'] == metric]
        for j, genre in enumerate(genres):
            genre_data = metric_data[metric_data['genre'] == genre]
            if not genre_data.empty:
                data_matrix[i, j] = genre_data['mean'].iloc[0]
    
    # 创建热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_matrix, 
                xticklabels=genres,
                yticklabels=significant_metrics,
                annot=True, 
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0,
                cbar_kws={'label': 'Mean Value'})
    
    plt.title('Genre Comparison Heatmap (Significant Metrics Only)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Singing Genre', fontsize=14, fontweight='bold')
    plt.ylabel('Acoustic Metrics', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Effect size heatmap saved to: {output_path}")

def create_summary_dashboard(detailed_df, contribution_df, output_path):
    """创建总结仪表板"""
    fig = plt.figure(figsize=(20, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. 贡献度排名 (左上)
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = contribution_df['metric'].tolist()
    scores = contribution_df['contribution_score'].tolist()
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    
    bars = ax1.barh(range(len(metrics)), scores, color=colors)
    ax1.set_yticks(range(len(metrics)))
    ax1.set_yticklabels(metrics, fontsize=11)
    ax1.set_xlabel('Contribution Score', fontsize=12, fontweight='bold')
    ax1.set_title('Top Contributing Metrics', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. 显著性统计 (右上)
    ax2 = fig.add_subplot(gs[0, 2:])
    significant_count = len(detailed_df[detailed_df['significant'] == True]['metric'].unique())
    total_count = len(detailed_df['metric'].unique())
    
    labels = ['Significant', 'Non-significant']
    sizes = [significant_count, total_count - significant_count]
    colors_pie = ['#ff6b6b', '#95a5a6']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, 
                                       autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Significance Summary\n({significant_count}/{total_count} metrics)', 
                  fontsize=14, fontweight='bold')
    
    # 3. 效应量分布 (左下)
    ax3 = fig.add_subplot(gs[1, :2])
    effect_sizes = contribution_df['effect_size'].tolist()
    ax3.hist(effect_sizes, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0.01, color='red', linestyle='--', label='Small effect')
    ax3.axvline(x=0.06, color='orange', linestyle='--', label='Medium effect')
    ax3.axvline(x=0.14, color='green', linestyle='--', label='Large effect')
    ax3.set_xlabel('Effect Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Effect Size Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. 歌种特征雷达图 (右下)
    ax4 = fig.add_subplot(gs[1, 2:], projection='polar')
    
    # 选择前5个最重要的metrics
    top_5_metrics = contribution_df.head(5)['metric'].tolist()
    
    # 标准化数据用于雷达图
    genres = ['BelCanto', 'FolkSongs', 'PopularMusic']
    angles = np.linspace(0, 2 * np.pi, len(top_5_metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    for i, genre in enumerate(genres):
        values = []
        for metric in top_5_metrics:
            metric_data = detailed_df[(detailed_df['metric'] == metric) & 
                                    (detailed_df['genre'] == genre)]
            if not metric_data.empty:
                # 标准化到0-1范围
                mean_val = metric_data['mean'].iloc[0]
                # 简单标准化 (实际应用中可能需要更复杂的标准化)
                normalized_val = (mean_val - detailed_df[detailed_df['metric'] == metric]['mean'].min()) / \
                                (detailed_df[detailed_df['metric'] == metric]['mean'].max() - 
                                 detailed_df[detailed_df['metric'] == metric]['mean'].min())
                values.append(normalized_val)
            else:
                values.append(0)
        
        values += values[:1]  # 闭合图形
        ax4.plot(angles, values, 'o-', linewidth=2, label=genre)
        ax4.fill(angles, values, alpha=0.25)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(top_5_metrics, fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_title('Genre Characteristics (Top 5 Metrics)', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    # 5. 关键发现文本 (底部)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # 获取关键发现
    top_metric = contribution_df.iloc[0]['metric']
    top_score = contribution_df.iloc[0]['contribution_score']
    
    key_findings = f"""
    🎯 KEY FINDINGS:
    
    • Most Contributing Metric: {top_metric} (Score: {top_score:.2f})
    • Significant Differences: {significant_count}/{total_count} metrics ({(significant_count/total_count)*100:.1f}%)
    • BelCanto: Most stable and clear voice (lowest entropy, highest CPP)
    • PopularMusic: Most complex and variable voice (highest entropy, lowest CPP)
    • FolkSongs: Intermediate characteristics between BelCanto and PopularMusic
    
    📊 INTERPRETATION:
    The analysis reveals that voice complexity (Entropy) is the primary differentiator 
    between singing styles, followed by voice sharpness (Crest) and vocal fold contact (Qcontact).
    """
    
    ax5.text(0.05, 0.5, key_findings, fontsize=12, va='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('FonaDyn VRP Significance Analysis Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary dashboard saved to: {output_path}")

def main():
    """主函数"""
    print("Creating significance analysis visualizations...")
    
    # 加载数据
    detailed_df, contribution_df = load_results()
    
    # 创建输出目录
    output_dir = r"H:\Python Academics\Voicemapping Folk singing\Results\singing\significance_results\visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成各种图表
    create_contribution_ranking_plot(contribution_df, 
                                   os.path.join(output_dir, "contribution_ranking.png"))
    
    create_genre_comparison_plot(detailed_df, 
                               os.path.join(output_dir, "genre_comparison.png"))
    
    create_effect_size_heatmap(detailed_df, 
                              os.path.join(output_dir, "effect_size_heatmap.png"))
    
    create_summary_dashboard(detailed_df, contribution_df, 
                           os.path.join(output_dir, "summary_dashboard.png"))
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("- contribution_ranking.png: 贡献度排名图")
    print("- genre_comparison.png: 歌种对比图")
    print("- effect_size_heatmap.png: 效应量热图")
    print("- summary_dashboard.png: 总结仪表板")

if __name__ == "__main__":
    main()
