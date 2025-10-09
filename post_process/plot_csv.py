# load two csv file and plot the data and the overlay

import matplotlib.pyplot as plt
import numpy as np
import sys
import csv

# the path of the folder
path1 = r'L:\Huanchen\Thyrovoice\audio\AMOlil\AMOlilpre_VRP.csv'
path2 = r'L:\Huanchen\Thyrovoice\audio\AMOlil\AMOlilpos_VRP.csv'
# path1 = '/Volumes/voicelab/Huanchen/Thyrovoice/audio/AMOlil/AMOlilpre_VRP.csv'
# path2 = '/Volumes/voicelab/Huanchen/Thyrovoice/audio/AMOlil/AMOlilpos_VRP.csv'

# read the csv files and save the data in the list and remove the header, delimiter is ';'
with open(path1, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    data1 = list(reader)
    data1.pop(0)

with open(path2, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    data2 = list(reader)
    data2.pop(0)

# Convert the lists to NumPy arrays for easier slicing
data1 = np.array(data1, dtype=float)
data2 = np.array(data2, dtype=float)

# data3 is the overlap of the first two cols of data1 and data2, when [i,j] match data1 and data2, save it in data3
data3 = []
for i in range(120):
    for j in range(120):
        if np.any(np.all(data1[:, 0:2] == [i, j], axis=1)) and np.any(np.all(data2[:, 0:2] == [i, j], axis=1)):
            data3.append([i,j])

# zero column is the x axis, first column is the y axis
# plot the first two colomns of the data and the overlaps, the type is 'o' and the color is 'blue'
plt.figure(figsize=(10, 5))
plt.plot(data1[:, 0], data1[:, 1], '1', color='blue')
plt.plot(data2[:, 0], data2[:, 1], '+', color='red')
# plot the overlap data3 with rectangle and the color is 'gray'
plt.plot(np.array(data3)[:, 0], np.array(data3)[:, 1], 's', color='gray')

# set the grid
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# set the label of the x axis and y axis
plt.xlabel('semitone')
plt.ylabel('dB')

# set the title
plt.title('Voice Map pre and post')

# set the legend
plt.legend(['pre', 'post', 'overlap'])

# show the plot
plt.show()