"""
将完整VRP结果转换为标准VRP CSV格式
"""

import pandas as pd
import numpy as np

def convert_to_standard_vrp(csv_file):
    """转换为标准VRP格式"""
    # 读取数据
    df = pd.read_csv(csv_file)
    
    print(f"=== 转换为标准VRP格式 ===")
    print(f"原始数据点数量: {len(df)}")
    
    # 创建标准VRP格式 - 完全匹配参考文件格式
    vrp_data = []
    
    # 按MIDI和SPL分组
    for midi in range(30, 100):
        for spl in range(30, 120):
            # 找到该bin内的所有数据点
            mask = (df['MIDI'] == midi) & (df['SPL'] == spl)
            bin_data = df[mask]
            
            if len(bin_data) > 0:
                # 计算该bin的统计值
                total = len(bin_data)
                clarity = bin_data['Clarity'].mean()
                cpp = bin_data['CPP'].mean()
                specbal = bin_data['SpectrumBalance'].mean()
                crest = bin_data['CrestFactor'].mean()
                cse = bin_data['CSE'].mean()  # 使用CSE作为Entropy
                qci = bin_data['QCI'].mean()
                deggmax = bin_data['dEGGmax'].mean()
                
                vrp_data.append({
                    'MIDI': midi,
                    'dB': spl,
                    'Total': total,
                    'Clarity': clarity,
                    'Crest': crest,
                    'SpecBal': specbal,
                    'CPP': cpp,
                    'Entropy': cse,  # 使用CSE作为Entropy
                    'dEGGmax': deggmax,
                    'Qcontact': qci,  # 使用QCI作为Qcontact
                    'Icontact': 0.0,  # 补零
                    'HRFegg': 0.0,    # 补零
                    'maxCluster': 1,   # 补零
                    'Cluster 1': total,  # 使用Total作为Cluster 1
                    'Cluster 2': 0.0,    # 补零
                    'Cluster 3': 0.0,    # 补零
                    'Cluster 4': 0.0,    # 补零
                    'Cluster 5': 0.0,    # 补零
                    'maxCPhon': 4,       # 补零
                    'cPhon 1': 0.0,      # 补零
                    'cPhon 2': 0.0,      # 补零
                    'cPhon 3': 0.0,      # 补零
                    'cPhon 4': total,    # 使用Total作为cPhon 4
                    'cPhon 5': 0.0       # 补零
                })
    
    # 创建DataFrame
    vrp_df = pd.DataFrame(vrp_data)
    
    # 保存标准VRP格式 - 使用分号分隔符
    output_file = "standard_vrp.csv"
    vrp_df.to_csv(output_file, index=False, sep=';')
    print(f"标准VRP格式已保存到: {output_file}")
    
    print(f"\n=== 标准VRP统计 ===")
    print(f"VRP数据点数量: {len(vrp_df)}")
    print(f"MIDI范围: {vrp_df['MIDI'].min()} - {vrp_df['MIDI'].max()}")
    print(f"SPL范围: {vrp_df['dB'].min()} - {vrp_df['dB'].max()}")
    
    print(f"\n列头信息:")
    print(f"总列数: {len(vrp_df.columns)}")
    print(f"列名: {list(vrp_df.columns)}")
    
    print(f"\n前5行数据:")
    print(vrp_df.head(5).to_string(index=False))
    
    return vrp_df

if __name__ == "__main__":
    convert_to_standard_vrp("complete_vrp_results.csv")
