# FonaDyn VRP 分析任务说明

## 📋 任务分工

### Task1: 统计分析
**文件**: `task1_statistics_analysis.py`
**功能**: 计算所有参数的统计指标（min, max, mean, std）
**输出位置**: `H:\Python Academics\Voicemapping Folk singing\Results\singing\statistics_results\`
**输出内容**: 
- `individual_statistics.csv` - 个人统计
- `genre_statistics.csv` - 歌种合并统计
- `gender_genre_statistics.csv` - 性别+歌种合并统计
- `statistics_summary.txt` - 汇总报告
**特点**: 使用Total作为权重进行加权统计

### Task2: 个人VRP图生成
**文件**: `task2_plot.py`
**功能**: 为每个人的CSV文件生成单独的VRP图
**输出位置**: `H:\Python Academics\Voicemapping Folk singing\Results\singing\result\`
**输出内容**: 32个个人VRP图
- 美声唱法: 11人
- 民歌: 10人  
- 流行音乐: 11人

### Task3: 歌种合并VRP图生成
**文件**: `task3_genre_only.py`
**功能**: 按歌种合并所有人员数据，生成歌种级别的VRP图
**输出位置**: `H:\Python Academics\Voicemapping Folk singing\Results\singing\genre_results\`
**输出内容**: 3个歌种合并VRP图
- `BelCanto_merged.png` - 美声唱法 (11人合并)
- `FolkSongs_merged.png` - 民歌 (10人合并)
- `PopularMusic_merged.png` - 流行音乐 (11人合并)

### Task4: 性别+歌种合并VRP图生成
**文件**: `task4_gender_genre_merge.py`
**功能**: 按性别+歌种组合合并数据，生成细分组合的VRP图
**输出位置**: `H:\Python Academics\Voicemapping Folk singing\Results\singing\gender_genre_results\`
**输出内容**: 6个性别+歌种组合VRP图
- `male_BelCanto_merged.png` - 男性美声唱法
- `female_BelCanto_merged.png` - 女性美声唱法
- `male_FolkSongs_merged.png` - 男性民歌
- `female_FolkSongs_merged.png` - 女性民歌
- `male_PopularMusic_merged.png` - 男性流行音乐
- `female_PopularMusic_merged.png` - 女性流行音乐

### Task5: 显著性差异分析
**文件**: `task5_significance_analysis.py`
**功能**: 分析三种唱法之间的显著性差异和贡献度
**输出位置**: `H:\Python Academics\Voicemapping Folk singing\Results\singing\significance_results\`
**输出内容**: 
- `significance_detailed_results.csv` - 详细显著性检验结果
- `contribution_analysis.csv` - 各metric的贡献度分析
- `significance_analysis_report.txt` - 综合分析报告
**特点**: 
- 使用ANOVA或Kruskal-Wallis检验
- 计算效应量和贡献度评分
- 识别造成差异的主要因素

## 🎯 分析层次

1. **统计分析** (Task1): 计算所有参数的统计指标
2. **个人层面** (Task2): 分析个体差异
3. **歌种层面** (Task3): 分析歌种特征
4. **性别+歌种层面** (Task4): 分析性别和歌种的交互影响
5. **显著性分析** (Task5): 分析三种唱法间的显著性差异和贡献度

## ✅ 所有VRP图特性

- 1.5:1横向拉长比例
- Threshold = 5 cycles 信息显示
- 9个子图完整声学指标
- 高质量数据过滤

## 🚫 任务边界

- **Task1**: 只进行统计分析，不生成图表
- **Task2**: 只生成个人图，不生成合并图
- **Task3**: 只生成歌种合并图，不生成个人图
- **Task4**: 只生成性别+歌种合并图，不生成个人图
- **Task5**: 只进行显著性分析，不生成图表
