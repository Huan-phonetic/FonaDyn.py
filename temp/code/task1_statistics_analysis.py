import os
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
import re

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
# 2. 按歌种和性别分类文件
# =========================================
def categorize_files(folder_path):
    """根据文件名按歌种和性别分类CSV文件"""
    csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    
    # 按歌种分类
    genre_files = defaultdict(list)
    # 按性别+歌种分类
    gender_genre_files = defaultdict(list)
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        # 跳过合并文件
        if "_merged.csv" in filename:
            continue
        
        # 提取歌种信息
        genre = extract_genre_from_path(filepath)
        genre_files[genre].append(filepath)
        
        # 提取性别和歌种信息
        gender, genre_gender = extract_gender_genre_from_path(filepath)
        if gender != "unknown":
            key = f"{gender}_{genre_gender}"
            gender_genre_files[key].append(filepath)
    
    return genre_files, gender_genre_files

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

def extract_gender_genre_from_path(filepath):
    """从文件路径提取性别和歌种信息"""
    filename = os.path.basename(filepath)
    
    # 提取性别信息
    gender = "unknown"
    if "_female_" in filename.lower():
        gender = "female"
    elif "_male_" in filename.lower():
        gender = "male"
    
    # 提取歌种信息
    genre = extract_genre_from_path(filepath)
    
    return gender, genre

# =========================================
# 3. 计算加权统计指标
# =========================================
def calculate_weighted_statistics(df, weight_col='Total'):
    """计算加权统计指标"""
    if df.empty or weight_col not in df.columns:
        return {}
    
    # 获取权重
    weights = df[weight_col].values
    total_weight = weights.sum()
    
    if total_weight == 0:
        return {}
    
    # 定义需要统计的列（排除maxCluster之后的列）
    target_columns = []
    for col in df.columns:
        if col == weight_col:
            continue
        if col == 'maxCluster':
            break  # 停止在maxCluster，不包含它及之后的列
        if df[col].dtype in ['int64', 'float64']:
            target_columns.append(col)
    
    stats = {}
    
    # 对每个目标列计算加权统计
    for col in target_columns:
        values = df[col].values
        
        # 加权统计
        weighted_mean = np.average(values, weights=weights)
        weighted_var = np.average((values - weighted_mean)**2, weights=weights)
        weighted_std = np.sqrt(weighted_var)
        
        stats[col] = {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(weighted_mean),
            'std': float(weighted_std)
        }
    
    return stats

# =========================================
# 4. 分析单个文件
# =========================================
def analyze_individual_file(filepath):
    """分析单个CSV文件"""
    df = load_csv_safe(filepath)
    if df is None or df.empty:
        return None
    
    filename = os.path.basename(filepath)
    gender, genre = extract_gender_genre_from_path(filepath)
    
    stats = calculate_weighted_statistics(df)
    
    result = {
        'filename': filename,
        'gender': gender,
        'genre': genre,
        'total_records': len(df),
        'statistics': stats
    }
    
    return result

# =========================================
# 5. 分析合并数据
# =========================================
def analyze_merged_data(file_paths, category_name):
    """分析合并后的数据"""
    all_data = []
    
    for filepath in file_paths:
        df = load_csv_safe(filepath)
        if df is not None and not df.empty:
            all_data.append(df)
    
    if not all_data:
        return None
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 按MIDI和dB分组，计算加权平均
    grouped = combined_df.groupby(['MIDI', 'dB'])
    merged_rows = []
    
    for (midi, db), group in grouped:
        if 'Total' not in group.columns:
            continue
            
        weights = group['Total'].values
        total_weight = weights.sum()
        
        if total_weight == 0:
            continue
        
        # 计算加权平均
        merged_row = {'MIDI': midi, 'dB': db}
        for col in group.columns:
            if col in ['MIDI', 'dB']:
                continue
            
            if group[col].dtype in ['int64', 'float64']:
                merged_row[col] = np.average(group[col].values, weights=weights)
            else:
                merged_row[col] = group[col].iloc[0]
        
        merged_rows.append(merged_row)
    
    merged_df = pd.DataFrame(merged_rows)
    
    if merged_df.empty:
        return None
    
    stats = calculate_weighted_statistics(merged_df)
    
    result = {
        'category': category_name,
        'total_files': len(file_paths),
        'total_records': len(merged_df),
        'statistics': stats
    }
    
    return result

# =========================================
# 6. 生成统计报告
# =========================================
def generate_statistics_report(folder_path, output_dir):
    """生成完整的统计报告"""
    print("Starting comprehensive statistics analysis...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 分类文件
    print("Categorizing files...")
    genre_files, gender_genre_files = categorize_files(folder_path)
    
    # 1. 个人统计
    print("\n1. Analyzing individual files...")
    individual_results = []
    csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    
    for filepath in csv_files:
        if "_merged.csv" in os.path.basename(filepath):
            continue
        
        result = analyze_individual_file(filepath)
        if result is not None:
            individual_results.append(result)
    
    # 2. 歌种合并统计
    print("\n2. Analyzing genre-merged data...")
    genre_results = []
    for genre, file_paths in genre_files.items():
        if genre == "unknown":
            continue
        
        result = analyze_merged_data(file_paths, f"Genre_{genre}")
        if result is not None:
            genre_results.append(result)
    
    # 3. 性别+歌种合并统计
    print("\n3. Analyzing gender-genre-merged data...")
    gender_genre_results = []
    for category, file_paths in gender_genre_files.items():
        result = analyze_merged_data(file_paths, f"GenderGenre_{category}")
        if result is not None:
            gender_genre_results.append(result)
    
    # 保存结果
    print("\n4. Saving results...")
    
    # 保存个人统计 - 展开格式
    individual_expanded = []
    for result in individual_results:
        base_info = {
            'filename': result['filename'],
            'gender': result['gender'],
            'genre': result['genre'],
            'total_records': result['total_records']
        }
        
        for metric, stats in result['statistics'].items():
            row = base_info.copy()
            row['metric'] = metric
            row['min'] = stats['min']
            row['max'] = stats['max']
            row['mean'] = stats['mean']
            row['std'] = stats['std']
            individual_expanded.append(row)
    
    individual_df = pd.DataFrame(individual_expanded)
    individual_file = os.path.join(output_dir, "individual_statistics.csv")
    individual_df.to_csv(individual_file, index=False, sep=';')
    print(f"Saved individual statistics: {individual_file}")
    
    # 保存歌种统计 - 展开格式
    genre_expanded = []
    for result in genre_results:
        base_info = {
            'category': result['category'],
            'total_files': result['total_files'],
            'total_records': result['total_records']
        }
        
        for metric, stats in result['statistics'].items():
            row = base_info.copy()
            row['metric'] = metric
            row['min'] = stats['min']
            row['max'] = stats['max']
            row['mean'] = stats['mean']
            row['std'] = stats['std']
            genre_expanded.append(row)
    
    genre_df = pd.DataFrame(genre_expanded)
    genre_file = os.path.join(output_dir, "genre_statistics.csv")
    genre_df.to_csv(genre_file, index=False, sep=';')
    print(f"Saved genre statistics: {genre_file}")
    
    # 保存性别+歌种统计 - 展开格式
    gender_genre_expanded = []
    for result in gender_genre_results:
        base_info = {
            'category': result['category'],
            'total_files': result['total_files'],
            'total_records': result['total_records']
        }
        
        for metric, stats in result['statistics'].items():
            row = base_info.copy()
            row['metric'] = metric
            row['min'] = stats['min']
            row['max'] = stats['max']
            row['mean'] = stats['mean']
            row['std'] = stats['std']
            gender_genre_expanded.append(row)
    
    gender_genre_df = pd.DataFrame(gender_genre_expanded)
    gender_genre_file = os.path.join(output_dir, "gender_genre_statistics.csv")
    gender_genre_df.to_csv(gender_genre_file, index=False, sep=';')
    print(f"Saved gender-genre statistics: {gender_genre_file}")
    
    # 生成汇总报告
    generate_summary_report(individual_results, genre_results, gender_genre_results, output_dir)
    
    print(f"\nAll statistics saved in: {output_dir}")

def generate_summary_report(individual_results, genre_results, gender_genre_results, output_dir):
    """生成汇总报告"""
    report_file = os.path.join(output_dir, "statistics_summary.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("FonaDyn VRP Statistics Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 列出所有metric
        f.write("METRICS ANALYZED:\n")
        f.write("-" * 20 + "\n")
        metrics = [
            "MIDI - MIDI pitch values (32-91)",
            "dB - Sound pressure level (30-120 dB)",
            "Total - Number of cycles for each (MIDI, dB) pair",
            "Clarity - Voice clarity measure (0.96-1.0)",
            "Crest - Crest factor (1.414-4)",
            "SpecBal - Spectral balance (-40 to -10 dB)",
            "CPP - Cepstral peak prominence (0-20)",
            "Entropy - Sample entropy (0-20)",
            "dEGGmax - Maximum derivative of EGG signal (0-1, log10 scale)",
            "Qcontact - Contact quotient (0.1-0.6)",
            "Icontact - Intensity of contact (0-0.6)"
        ]
        for metric in metrics:
            f.write(f"  • {metric}\n")
        
        f.write("\n\nSTATISTICS CALCULATED FOR EACH METRIC:\n")
        f.write("-" * 40 + "\n")
        f.write("  • min - Minimum value\n")
        f.write("  • max - Maximum value\n")
        f.write("  • mean - Weighted mean (using Total as weight)\n")
        f.write("  • std - Weighted standard deviation\n")
        f.write("  • count - Number of data points\n")
        f.write("  • total_weight - Sum of Total values\n")
        
        # 个人统计汇总
        f.write("\n\n1. INDIVIDUAL STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total individuals analyzed: {len(individual_results)}\n")
        
        # 按歌种分组
        genre_counts = defaultdict(int)
        gender_counts = defaultdict(int)
        for result in individual_results:
            genre_counts[result['genre']] += 1
            gender_counts[result['gender']] += 1
        
        f.write("\nBy Genre:\n")
        for genre, count in genre_counts.items():
            f.write(f"  {genre}: {count} individuals\n")
        
        f.write("\nBy Gender:\n")
        for gender, count in gender_counts.items():
            f.write(f"  {gender}: {count} individuals\n")
        
        # 歌种统计汇总
        f.write("\n\n2. GENRE-MERGED STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total genres analyzed: {len(genre_results)}\n")
        for result in genre_results:
            f.write(f"  {result['category']}: {result['total_files']} files, {result['total_records']} records\n")
        
        # 性别+歌种统计汇总
        f.write("\n\n3. GENDER-GENRE-MERGED STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total gender-genre combinations: {len(gender_genre_results)}\n")
        for result in gender_genre_results:
            f.write(f"  {result['category']}: {result['total_files']} files, {result['total_records']} records\n")
        
        f.write("\n\nReport generated successfully!\n")
    
    print(f"Saved summary report: {report_file}")

# =========================================
# 7. 主入口
# =========================================
if __name__ == "__main__":
    # 指定要处理的文件夹路径
    target_folder = r"H:\Python Academics\Voicemapping Folk singing\Results\singing"
    output_folder = r"H:\Python Academics\Voicemapping Folk singing\Results\singing\statistics_final"
    
    # 检查文件夹是否存在
    if not os.path.exists(target_folder):
        print(f"Error: Folder not found - {target_folder}")
        print("Please check if the path is correct")
    else:
        print(f"Processing folder: {target_folder}")
        generate_statistics_report(target_folder, output_folder)
        print("Task1 (Statistics Analysis) completed!")