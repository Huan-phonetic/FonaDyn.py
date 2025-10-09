"""
Input: A folder containing multiple CSV files with the same structure
Output: An Excel file containing statistics for each CSV file
"""


import os
import pandas as pd

# Function to calculate statistics for a given dataframe
def calculate_statistics(df, file_name):
    stats = {}
    stats['File'] = file_name
    stats['Total Items'] = len(df)
    stats['Sum of Total'] = df['Total'].sum()

    for col in ['MIDI', 'dB', 'Clarity', 'Crest', 'SpecBal', 'CPP']:
        stats[f'{col} Min'] = df[col].min()
        stats[f'{col} Mean'] = df[col].mean()
        stats[f'{col} Max'] = df[col].max()
        stats[f'{col} Std'] = df[col].std()

    return stats

# Directory containing CSV files
folder = r'F:\Coqui\TTS\resampled_output\results\sentence_1'

# List to hold the statistics for all files
all_stats = []

# Iterate over all CSV files in the folder
for file in os.listdir(folder):
    if file.endswith('.csv'):
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path, delimiter=';')
        file_stats = calculate_statistics(df, file)
        all_stats.append(file_stats)

# Create a DataFrame from the collected statistics
stats_df = pd.DataFrame(all_stats)

# Write the statistics to an Excel file
output_file = 'sentence_1_stats.xlsx'
output_file = os.path.join(folder, output_file)
stats_df.to_excel(output_file, index=False)

print(f'Statistics summary has been written to {output_file}')

def loadVRP(folder_path, containKeywords=None):
    # Load all '_VRP.csv' files in the folder
    files = os.listdir(folder_path)
    if containKeywords:
        files = [f for f in files if any(k in f for k in containKeywords)]

    # Load the contents of the files, the csv files are separated by ';'
    data = {}
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            data[file] = lines
    return data

def computeMetrics(data):
    # Compute metrics from the data
    metrics = {}
    for file, lines in data.items():
        # Extract the metrics from the lines
        # The fifth col is Crest, the sixth col is SpecBal, the seventh col is CPPs in natural order
        # compute the min, max, and average, p-value, confidence interval
        # The first row is the header
        crest = [float(line.split(';')[4]) for line in lines[1:]]
        specbal = [float(line.split(';')[5]) for line in lines[1:]]
        cpps = [float(line.split(';')[6]) for line in lines[1:]]
        metrics[file] = {
            'crest': {
                'min': min(crest),
                'max': max(crest),
                'average': sum(crest) / len(crest)
            },
            'specbal': {
                'min': min(specbal),
                'max': max(specbal),
                'average': sum(specbal) / len(specbal)
            },
            'cpps': {
                'min': min(cpps),
                'max': max(cpps),
                'average': sum(cpps) / len(cpps)
            }
        }
    return metrics