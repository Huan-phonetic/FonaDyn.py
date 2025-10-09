import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = '/Volumes/voicelab/Huanchen/TTS/results/sentence_100/sentence_100_stats.xlsx'
data = pd.read_excel(file_path)

# Extract relevant metrics
metrics = {
    'dB': ['dB Min', 'dB Mean', 'dB Max', 'dB Std'],
    'MIDI': ['MIDI Min', 'MIDI Mean', 'MIDI Max', 'MIDI Std'],
    'Crest': ['Crest Min', 'Crest Mean', 'Crest Max', 'Crest Std'],
    'SpecBal': ['SpecBal Min', 'SpecBal Mean', 'SpecBal Max', 'SpecBal Std'],
    'CPP': ['CPP Min', 'CPP Mean', 'CPP Max', 'CPP Std']
}

# Create a figure with subplots for each metric
fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 25))

for i, (metric, columns) in enumerate(metrics.items()):
    # Combine the data for the current metric into a single DataFrame
    combined_data = data.melt(id_vars=['File'], value_vars=columns, var_name='Metric', value_name='Value')
    sns.boxplot(data=combined_data, x='File', y='Value', ax=axes[i])
    axes[i].set_title(f'Distribution of {metric} metrics')
    axes[i].set_xlabel('Model')
    axes[i].set_ylabel(f'{metric} Value')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
