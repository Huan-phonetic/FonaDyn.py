import os
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
import re

# 导入task2_plot的绘图函数
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from task2_plot import plot_vrp, load_vrp

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
# 2. 按性别和歌种分类文件
# =========================================
def categorize_files_by_gender_genre(folder_path):
    """根据文件名按性别和歌种分类CSV文件"""
    csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    
    # 按性别+歌种分类的字典
    gender_genre_files = defaultdict(list)
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        folder_name = os.path.basename(os.path.dirname(filepath))
        
        # 跳过合并文件（包含_merged.csv的文件）
        if "_merged.csv" in filename:
            print(f"Skipping merged file: {filename}")
            continue
        
        # 提取性别和歌种信息
        gender, genre = extract_gender_genre_from_path(filepath)
        
        # 跳过无法识别性别的文件
        if gender == "unknown":
            print(f"Skipping file with unknown gender: {filename}")
            continue
            
        key = f"{gender}_{genre}"
        gender_genre_files[key].append(filepath)
    
    return gender_genre_files

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
    genre = "unknown"
    
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
                genre = genre_name
                break
        if genre != "unknown":
            break
    
    return gender, genre

# =========================================
# 3. 加权平均合并
# =========================================
def weighted_average_merge(dataframes):
    """以Total为权重合并多个DataFrame"""
    if not dataframes:
        return pd.DataFrame()
    
    # 合并所有数据
    all_data = []
    for df in dataframes:
        if df is not None and not df.empty:
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 按MIDI和dB分组
    grouped = combined_df.groupby(['MIDI', 'dB'])
    
    # 计算加权平均
    result_rows = []
    
    for (midi, db), group in grouped:
        if 'Total' not in group.columns:
            continue
            
        # 计算权重（Total值）
        weights = group['Total'].values
        total_weight = weights.sum()
        
        if total_weight == 0:
            continue
        
        # 计算加权平均
        weighted_avg = {}
        weighted_avg['MIDI'] = midi
        weighted_avg['dB'] = db
        
        # 对每个数值列计算加权平均
        for col in group.columns:
            if col in ['MIDI', 'dB']:
                continue
            
            if group[col].dtype in ['int64', 'float64']:
                # 数值列：加权平均
                weighted_avg[col] = np.average(group[col].values, weights=weights)
            else:
                # 非数值列：取第一个值
                weighted_avg[col] = group[col].iloc[0]
        
        result_rows.append(weighted_avg)
    
    result_df = pd.DataFrame(result_rows)
    return result_df

# =========================================
# 4. 处理单个性别+歌种组合
# =========================================
def process_gender_genre(gender, genre, file_paths, output_dir):
    """处理单个性别+歌种组合的所有CSV文件"""
    category_name = f"{gender}_{genre}"
    print(f"Processing category: {category_name}")
    print(f"Found {len(file_paths)} files")
    
    # 读取所有CSV文件
    dataframes = []
    for filepath in file_paths:
        print(f"  Loading: {os.path.basename(filepath)}")
        df = load_csv_safe(filepath)
        if df is not None:
            dataframes.append(df)
    
    if not dataframes:
        print(f"  No valid data found for {category_name}")
        return None
    
    # 合并数据
    print(f"  Merging {len(dataframes)} dataframes...")
    merged_df = weighted_average_merge(dataframes)
    
    if merged_df.empty:
        print(f"  No data after merging for {category_name}")
        return None
    
    # 保存合并后的CSV
    output_file = os.path.join(output_dir, f"{category_name}_merged.csv")
    merged_df.to_csv(output_file, index=False, sep=';')
    print(f"  Saved merged data: {output_file}")
    print(f"  Merged data shape: {merged_df.shape}")
    
    return merged_df, output_file

# =========================================
# 5. 主处理函数
# =========================================
def process_all_gender_genre_combinations(folder_path, output_dir):
    """处理所有性别+歌种组合"""
    print("Starting gender-genre based CSV merging and plotting...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 按性别+歌种分类文件
    print("Categorizing files by gender and genre...")
    gender_genre_files = categorize_files_by_gender_genre(folder_path)
    
    if not gender_genre_files:
        print("No CSV files found!")
        return
    
    print(f"Found {len(gender_genre_files)} gender-genre combinations:")
    for category, files in gender_genre_files.items():
        print(f"  {category}: {len(files)} files")
        for file in files[:2]:  # 只显示前2个文件作为示例
            print(f"    - {os.path.basename(file)}")
        if len(files) > 2:
            print(f"    ... and {len(files)-2} more files")
    
    # 处理每个性别+歌种组合
    merged_files = []
    for category_name, file_paths in gender_genre_files.items():
        # 解析类别名称
        parts = category_name.split('_')
        if len(parts) >= 2:
            gender = parts[0]
            genre = '_'.join(parts[1:])  # 处理可能包含下划线的歌种名
        else:
            print(f"Warning: Cannot parse category name: {category_name}")
            continue
            
        result = process_gender_genre(gender, genre, file_paths, output_dir)
        if result is not None:
            merged_df, output_file = result
            merged_files.append((category_name, output_file))
    
    # 使用task2_plot为每个性别+歌种组合绘图
    print("\nGenerating plots for each gender-genre combination...")
    for category_name, csv_file in merged_files:
        print(f"Plotting {category_name}...")
        try:
            df = load_vrp(csv_file)
            plot_vrp(df, csv_file, output_dir)
        except Exception as e:
            print(f"Error plotting {category_name}: {e}")
    
    print(f"\nAll done! Results saved in: {output_dir}")

# =========================================
# 6. 主入口
# =========================================
if __name__ == "__main__":
    # 指定要处理的文件夹路径
    target_folder = r"H:\Python Academics\Voicemapping Folk singing\Results\singing"
    output_folder = r"H:\Python Academics\Voicemapping Folk singing\Results\singing\gender_genre_results"
    
    # 检查文件夹是否存在
    if not os.path.exists(target_folder):
        print(f"Error: Folder not found - {target_folder}")
        print("Please check if the path is correct")
    else:
        print(f"Processing folder: {target_folder}")
        process_all_gender_genre_combinations(target_folder, output_folder)
        print("Task4 completed!")
