#!/usr/bin/env python3
"""
VoiceMap Usage Examples
Updated for new package structure
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analyzer import VoiceMapAnalyzer
from config import VoiceMapConfig
import os

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建分析器
    analyzer = VoiceMapAnalyzer()
    
    # 运行分析
    data, output_file = analyzer.analyze_and_output_vrp()
    
    print(f"分析完成！")
    print(f"输出文件: {output_file}")
    print(f"数据点数: {len(data['midi']):,}")
    print(f"MIDI范围: {data['midi'].min():.1f} - {data['midi'].max():.1f}")
    print(f"SPL范围: {data['spl'].min():.1f} - {data['spl'].max():.1f} dB")

def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    # 创建自定义配置
    config = VoiceMapConfig(
        clarity_threshold=0.95,      # 更严格的Clarity阈值
        spl_correction_db=120.0,     # SPL校正
        output_dir="custom_results"  # 自定义输出目录
    )
    
    # 创建分析器
    analyzer = FonaDynAnalyzer(config)
    
    # 运行分析
    data, output_file = analyzer.analyze_and_output_vrp()
    
    print(f"自定义分析完成！")
    print(f"输出文件: {output_file}")
    print(f"Clarity阈值: {config.clarity_threshold}")
    print(f"数据点数: {len(data['midi']):,}")

def example_custom_audio():
    """自定义音频文件示例"""
    print("\n=== 自定义音频文件示例 ===")
    
    # 检查音频文件是否存在
    audio_file = r"G:\FonaDyn.py\audio\test_Voice_EGG.wav"
    
    if os.path.exists(audio_file):
        analyzer = VoiceMapAnalyzer()
        data, output_file = analyzer.analyze_and_output_vrp(audio_file)
        
        print(f"音频文件分析完成！")
        print(f"音频文件: {audio_file}")
        print(f"输出文件: {output_file}")
        print(f"数据点数: {len(data['midi']):,}")
    else:
        print(f"音频文件不存在: {audio_file}")

def example_data_analysis():
    """数据分析示例"""
    print("\n=== 数据分析示例 ===")
    
    analyzer = VoiceMapAnalyzer()
    data, output_file = analyzer.analyze_and_output_vrp()
    
    # 分析各个指标
    print("各指标统计:")
    print(f"MIDI: {data['midi'].mean():.1f} ± {data['midi'].std():.1f}")
    print(f"SPL: {data['spl'].mean():.1f} ± {data['spl'].std():.1f} dB")
    print(f"Clarity: {data['clarity'].mean():.3f} ± {data['clarity'].std():.3f}")
    print(f"CPP: {data['cpp'].mean():.3f} ± {data['cpp'].std():.3f}")
    print(f"SpecBal: {data['specbal'].mean():.3f} ± {data['specbal'].std():.3f}")
    print(f"Crest: {data['crest'].mean():.3f} ± {data['crest'].std():.3f}")
    print(f"Qcontact: {data['qcontact'].mean():.3f} ± {data['qcontact'].std():.3f}")
    print(f"dEGGmax: {data['deggmax'].mean():.3f} ± {data['deggmax'].std():.3f}")
    print(f"Icontact: {data['icontact'].mean():.3f} ± {data['icontact'].std():.3f}")

if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_custom_config()
    example_custom_audio()
    example_data_analysis()