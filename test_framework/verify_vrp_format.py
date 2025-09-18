"""
验证标准VRP格式是否与参考文件完全一致
"""

import pandas as pd

def verify_vrp_format():
    """验证VRP格式"""
    print("=== 验证标准VRP格式 ===")
    
    # 读取参考文件
    ref_file = "../audio/test_Voice_EGG_VRP.csv"
    ref_df = pd.read_csv(ref_file, sep=';')
    
    # 读取我们的输出文件
    our_file = "standard_vrp.csv"
    our_df = pd.read_csv(our_file, sep=';')
    
    print(f"参考文件列数: {len(ref_df.columns)}")
    print(f"我们的文件列数: {len(our_df.columns)}")
    
    print(f"\n参考文件列名:")
    print(list(ref_df.columns))
    
    print(f"\n我们的文件列名:")
    print(list(our_df.columns))
    
    # 检查列名是否一致
    if list(ref_df.columns) == list(our_df.columns):
        print("\n✅ 列名完全一致！")
    else:
        print("\n❌ 列名不一致！")
        missing_cols = set(ref_df.columns) - set(our_df.columns)
        extra_cols = set(our_df.columns) - set(ref_df.columns)
        if missing_cols:
            print(f"缺失的列: {missing_cols}")
        if extra_cols:
            print(f"多余的列: {extra_cols}")
    
    # 检查数据类型
    print(f"\n参考文件数据类型:")
    print(ref_df.dtypes)
    
    print(f"\n我们的文件数据类型:")
    print(our_df.dtypes)
    
    # 显示样本数据
    print(f"\n参考文件前3行:")
    print(ref_df.head(3).to_string(index=False))
    
    print(f"\n我们的文件前3行:")
    print(our_df.head(3).to_string(index=False))
    
    print(f"\n=== 格式验证完成 ===")

if __name__ == "__main__":
    verify_vrp_format()
