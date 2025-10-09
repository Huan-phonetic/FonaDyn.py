import os
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
from scipy import stats
from scipy.stats import f_oneway, kruskal
import warnings
warnings.filterwarnings('ignore')

# =========================================
# 1. 读取CSV文件
# =========================================
def load_csv_safe(filepath):
    """安全读取CSV文件，自动检测分隔符"""
    try:
        df = pd.read_csv(filepath, sep=';', engine='python')
    except Exception:
        try:
            df = pd.read_csv(filepath, sep=',', engine='python')
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None
    
    # 清理列名
    df.columns = [c.strip() for c in df.columns]
    
    # 检查必要的列
    if "MIDI" not in df.columns or "dB" not in df.columns:
        print(f"Warning: {filepath} missing MIDI or dB columns")
        return None
    
    return df

# =========================================
# 2. 按歌种分类文件
# =========================================
def categorize_files_by_genre(folder_path):
    """根据文件名按歌种分类CSV文件"""
    csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    
    # 按歌种分类的字典
    genre_files = defaultdict(list)
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        # 跳过合并文件
        if "_merged.csv" in filename:
            continue
        
        # 提取歌种信息
        genre = extract_genre_from_path(filepath)
        if genre != "unknown":
            genre_files[genre].append(filepath)
    
    return genre_files

def extract_genre_from_path(filepath):
    """从文件路径提取歌种信息"""
    filename = os.path.basename(filepath)
    
    # 常见的歌种关键词
    genre_keywords = {
        'BelCanto': ['美声', 'belcanto', 'bel_canto'],
        'FolkSongs': ['民歌', 'folk', 'folk_songs'],
        'PopularMusic': ['通俗', 'pop', 'popular', '流行']
    }
    
    filename_lower = filename.lower()
    for genre_name, keywords in genre_keywords.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return genre_name
    
    return "unknown"

# =========================================
# 3. 数据预处理
# =========================================
def preprocess_data_for_analysis(genre_files):
    """预处理数据用于统计分析"""
    genre_data = {}
    
    for genre, file_paths in genre_files.items():
        all_data = []
        
        for filepath in file_paths:
            df = load_csv_safe(filepath)
            if df is not None and not df.empty:
                # 应用阈值过滤
                if 'Total' in df.columns:
                    df = df[df['Total'] >= 5]  # 使用相同的阈值
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            genre_data[genre] = combined_df
    
    return genre_data

# =========================================
# 4. 显著性检验
# =========================================
def perform_significance_tests(genre_data):
    """执行显著性检验"""
    # 定义需要分析的metrics（排除maxCluster之后）
    target_columns = []
    if genre_data:
        sample_df = list(genre_data.values())[0]
        for col in sample_df.columns:
            if col == 'maxCluster':
                break
            if col not in ['MIDI', 'dB', 'Total'] and sample_df[col].dtype in ['int64', 'float64']:
                target_columns.append(col)
    
    results = {}
    
    for metric in target_columns:
        print(f"Analyzing metric: {metric}")
        
        # 收集每个歌种的数据
        genre_values = {}
        for genre, df in genre_data.items():
            if metric in df.columns:
                values = df[metric].values
                # 移除异常值
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    genre_values[genre] = values
        
        if len(genre_values) >= 2:
            # 准备数据
            groups = list(genre_values.values())
            group_names = list(genre_values.keys())
            
            # 检查数据分布
            is_normal = []
            for group in groups:
                if len(group) > 3:
                    _, p_value = stats.shapiro(group[:5000])  # 限制样本大小
                    is_normal.append(p_value > 0.05)
                else:
                    is_normal.append(False)
            
            # 选择适当的检验方法
            if all(is_normal) and len(groups) == 3:
                # 正态分布且三组：使用ANOVA
                f_stat, p_value = f_oneway(*groups)
                test_type = "ANOVA"
            else:
                # 非正态分布或组数不同：使用Kruskal-Wallis
                h_stat, p_value = kruskal(*groups)
                test_type = "Kruskal-Wallis"
            
            # 计算效应量（eta-squared for ANOVA, epsilon-squared for Kruskal-Wallis）
            if test_type == "ANOVA":
                # Eta-squared
                ss_between = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(groups)))**2 for group in groups)
                ss_total = sum((np.concatenate(groups) - np.mean(np.concatenate(groups)))**2)
                effect_size = ss_between / ss_total if ss_total > 0 else 0
            else:
                # Epsilon-squared approximation
                n_total = sum(len(group) for group in groups)
                effect_size = (h_stat - len(groups) + 1) / (n_total - len(groups)) if n_total > len(groups) else 0
            
            # 计算描述性统计
            descriptive_stats = {}
            for genre, values in genre_values.items():
                descriptive_stats[genre] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'n': len(values)
                }
            
            results[metric] = {
                'test_type': test_type,
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < 0.05,
                'descriptive_stats': descriptive_stats,
                'group_names': group_names
            }
    
    return results

# =========================================
# 5. 贡献度分析
# =========================================
def calculate_contribution_analysis(genre_data, significance_results):
    """计算各metric的贡献度"""
    contributions = {}
    
    for metric, result in significance_results.items():
        if result['significant']:
            # 计算组间差异的贡献度
            effect_size = result['effect_size']
            p_value = result['p_value']
            
            # 综合评分：效应量权重70%，显著性权重30%
            significance_score = -np.log10(p_value) if p_value > 0 else 10
            contribution_score = effect_size * 0.7 + (significance_score / 10) * 0.3
            
            contributions[metric] = {
                'effect_size': effect_size,
                'p_value': p_value,
                'contribution_score': contribution_score,
                'significance_score': significance_score
            }
    
    # 按贡献度排序
    sorted_contributions = sorted(contributions.items(), 
                                key=lambda x: x[1]['contribution_score'], 
                                reverse=True)
    
    return sorted_contributions

# =========================================
# 6. 生成报告
# =========================================
def generate_significance_report(genre_data, significance_results, contributions, output_dir):
    """生成显著性分析报告"""
    
    # 1. 保存详细结果到CSV
    detailed_results = []
    for metric, result in significance_results.items():
        for genre, stats in result['descriptive_stats'].items():
            detailed_results.append({
                'metric': metric,
                'genre': genre,
                'mean': stats['mean'],
                'std': stats['std'],
                'median': stats['median'],
                'n': stats['n'],
                'test_type': result['test_type'],
                'p_value': result['p_value'],
                'effect_size': result['effect_size'],
                'significant': result['significant']
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_file = os.path.join(output_dir, "significance_detailed_results.csv")
    detailed_df.to_csv(detailed_file, index=False, sep=';')
    print(f"Saved detailed results: {detailed_file}")
    
    # 2. 保存贡献度分析
    contribution_results = []
    for metric, contrib in contributions:
        contribution_results.append({
            'metric': metric,
            'effect_size': contrib['effect_size'],
            'p_value': contrib['p_value'],
            'contribution_score': contrib['contribution_score'],
            'significance_score': contrib['significance_score']
        })
    
    contribution_df = pd.DataFrame(contribution_results)
    contribution_file = os.path.join(output_dir, "contribution_analysis.csv")
    contribution_df.to_csv(contribution_file, index=False, sep=';')
    print(f"Saved contribution analysis: {contribution_file}")
    
    # 3. 生成文本报告
    report_file = os.path.join(output_dir, "significance_analysis_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("FonaDyn VRP Significance Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 总体统计
        total_metrics = len(significance_results)
        significant_metrics = sum(1 for r in significance_results.values() if r['significant'])
        
        f.write("OVERALL SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total metrics analyzed: {total_metrics}\n")
        f.write(f"Significant differences found: {significant_metrics}\n")
        f.write(f"Significance rate: {significant_metrics/total_metrics*100:.1f}%\n\n")
        
        # 显著性结果
        f.write("SIGNIFICANCE TEST RESULTS\n")
        f.write("-" * 30 + "\n")
        for metric, result in significance_results.items():
            f.write(f"\n{metric}:\n")
            f.write(f"  Test: {result['test_type']}\n")
            f.write(f"  p-value: {result['p_value']:.6f}\n")
            f.write(f"  Effect size: {result['effect_size']:.4f}\n")
            f.write(f"  Significant: {'Yes' if result['significant'] else 'No'}\n")
            
            f.write("  Descriptive statistics:\n")
            for genre, stats in result['descriptive_stats'].items():
                f.write(f"    {genre}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}\n")
        
        # 贡献度分析
        f.write("\n\nCONTRIBUTION ANALYSIS\n")
        f.write("-" * 25 + "\n")
        f.write("Metrics ranked by contribution to genre differences:\n\n")
        
        for i, (metric, contrib) in enumerate(contributions, 1):
            f.write(f"{i}. {metric}\n")
            f.write(f"   Contribution Score: {contrib['contribution_score']:.4f}\n")
            f.write(f"   Effect Size: {contrib['effect_size']:.4f}\n")
            f.write(f"   p-value: {contrib['p_value']:.6f}\n")
            f.write(f"   Significance Score: {contrib['significance_score']:.2f}\n\n")
        
        # 主要发现
        f.write("KEY FINDINGS\n")
        f.write("-" * 15 + "\n")
        if contributions:
            top_metric = contributions[0][0]
            top_contribution = contributions[0][1]['contribution_score']
            f.write(f"• Most contributing metric: {top_metric} (score: {top_contribution:.4f})\n")
            f.write(f"• This metric explains {top_contribution*100:.1f}% of the variance between genres\n")
        
        significant_count = len(contributions)
        f.write(f"• {significant_count} metrics show significant differences between genres\n")
        
        if significant_count > 0:
            f.write("• Significant metrics (in order of importance):\n")
            for i, (metric, _) in enumerate(contributions[:5], 1):
                f.write(f"  {i}. {metric}\n")
        
        f.write("\nReport generated successfully!\n")
    
    print(f"Saved analysis report: {report_file}")

# =========================================
# 7. 主处理函数
# =========================================
def perform_significance_analysis(folder_path, output_dir):
    """执行完整的显著性分析"""
    print("Starting significance analysis...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 分类文件
    print("1. Categorizing files by genre...")
    genre_files = categorize_files_by_genre(folder_path)
    
    if len(genre_files) < 2:
        print("Error: Need at least 2 genres for comparison")
        return
    
    print(f"Found {len(genre_files)} genres: {list(genre_files.keys())}")
    
    # 2. 预处理数据
    print("2. Preprocessing data...")
    genre_data = preprocess_data_for_analysis(genre_files)
    
    # 3. 执行显著性检验
    print("3. Performing significance tests...")
    significance_results = perform_significance_tests(genre_data)
    
    # 4. 计算贡献度
    print("4. Calculating contribution analysis...")
    contributions = calculate_contribution_analysis(genre_data, significance_results)
    
    # 5. 生成报告
    print("5. Generating reports...")
    generate_significance_report(genre_data, significance_results, contributions, output_dir)
    
    print(f"\nAll results saved in: {output_dir}")

# =========================================
# 8. 主入口
# =========================================
if __name__ == "__main__":
    # 指定要处理的文件夹路径
    target_folder = r"H:\Python Academics\Voicemapping Folk singing\Results\singing"
    output_folder = r"H:\Python Academics\Voicemapping Folk singing\Results\singing\significance_results"
    
    # 检查文件夹是否存在
    if not os.path.exists(target_folder):
        print(f"Error: Folder not found - {target_folder}")
        print("Please check if the path is correct")
    else:
        print(f"Processing folder: {target_folder}")
        perform_significance_analysis(target_folder, output_folder)
        print("Task5 (Significance Analysis) completed!")
