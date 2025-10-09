import os
import pandas as pd

path = r'L:\Huanchen\Thyrovoice\Py_VRP\group_by_sur_type\group_by_all\Total'
union_path = r'L:\Huanchen\Thyrovoice\Py_VRP\group_by_sur_type\union'
# load all csv files in the path
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')] 
# output name is the two parts after the second last '\'
output_name = path.split('\\')[-2] + '_' + path.split('\\')[-1] + '.csv'
# in the list there are k=2 to k=5, need to create union csv of each k
for i in range(2,6):
    # match the csv file name, if the filename contains i=k and 'pre', then add it to the list pre_list
    pre_list = [f for f in csv_files if 'pre' in f and 'k='+str(i) in f]
    post_list = [f for f in csv_files if 'post' in f and 'k='+str(i) in f]
    # load all the csv files in the vrp_data
    pre_data = [pd.read_csv(os.path.join(path, pre)) for pre in pre_list]
    post_data = [pd.read_csv(os.path.join(path, post)) for post in post_list]
    # union the csv files
    pre_union = pd.concat(pre_data)
    post_union = pd.concat(post_data)
    # save the union csv files
    pre_union.to_csv(os.path.join(union_path, 'pre_k='+str(i) + '_'+ output_name), index=False)
    post_union.to_csv(os.path.join(union_path, 'post_k='+str(i) +'_'+ output_name), index=False)
