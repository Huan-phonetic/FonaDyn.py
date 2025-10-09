from read_csv import collect_VRP_data_v2, collect_VRP_data_v3
import numpy as np
import pandas as pd
import os
os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# path = '/Volumes/voicelab/Huanchen/Thyrovoice/bysubject_vrp_cycle5'
path = r'L:\Huanchen\Thyrovoice\audio_db_replaced_overlap_cluster_VRP'
VRP_path = r'L:\Huanchen\Thyrovoice\Py_db_VRP'
# VRP_path = '/Volumes/voicelab/Huanchen/Thyrovoice/Py_VRP'

# load the data
pre_VRP_data, post_VRP_data, name, _ = collect_VRP_data_v2(path)
metrics_names = [['MIDI', 'dB', 'Total', 'Crest', 'SpecBal', 'CPPs', 'Entropy', 'dEGGmax', 'Qcontact', 'maxCluster', 'Cluster 1', 'Cluster 2'],
                 ['MIDI', 'dB', 'Total', 'Crest', 'SpecBal', 'CPPs', 'Entropy', 'dEGGmax', 'Qcontact', 'maxCluster', 'Cluster 1', 'Cluster 2', 'Cluster 3'],
                    ['MIDI', 'dB', 'Total', 'Crest', 'SpecBal', 'CPPs', 'Entropy', 'dEGGmax', 'Qcontact', 'maxCluster', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'],
                    ['MIDI', 'dB', 'Total', 'Crest', 'SpecBal', 'CPPs', 'Entropy', 'dEGGmax', 'Qcontact', 'maxCluster', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']]
# the column names are: MIDI	dB	Total	Crest	SpecBal	CPPs	Entropy	dEGGmax	Qcontact	maxCluster	Cluster 1	Cluster 2
# Crest	SpecBal	CPPs	Entropy	dEGGmax	Qcontact:3 to 8
# pre process the data
# deggmax need to be log10, the dEGGmax column
for i in range(len(pre_VRP_data)):
    for j in range(len(pre_VRP_data[i])):
        pre_VRP_data[i][j][7] = np.log10(pre_VRP_data[i][j][7])
        post_VRP_data[i][j][7] = np.log10(post_VRP_data[i][j][7])

# 1. group by each subject, subject count X k centroids
def group_by_subject(pre, post, name, metrics_names=metrics_names):
    # create a folder to save the data named group_by_subject
    folder_name = 'group_by_subject'
    folder_path = os.path.join(VRP_path, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    centroids = []
    # each clustering only focuses on one subject, so normalize and normalize pre and post data by subject
    for i in range(len(pre)):
        # kmeans clustering
        for k in range(2, 6):
            pre_data = np.array(pre[i])
            post_data = np.array(post[i])
            # concatenate the pre and post data for each subject
            data = np.concatenate((pre_data[:, 3:9], post_data[:, 3:9]), axis=0)
            # standardize the data
            standard_scaler = StandardScaler()
            standard_scaler.fit(data)
            data = standard_scaler.transform(data)
            # normalize the data
            minmax_scaler = MinMaxScaler()
            minmax_scaler.fit(data)
            data = minmax_scaler.transform(data)
            # kmeans clustering
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=500, random_state=None)
            labels = kmeans.fit_predict(data)
            labels = reorder_clusters(np.concatenate((pre_data, post_data), axis=0), labels)
            #save the centroids
            centroid_name = name[i] + '_k=' + str(k)
            centroid = kmeans.cluster_centers_
            centroid = minmax_scaler.inverse_transform(centroid)
            centroid = standard_scaler.inverse_transform(centroid)
            centroids.append([centroid_name, centroid])
            # match the original index
            pre_data[:, 9] = labels[:len(pre_data)] + 1
            post_data[:, 9] = labels[len(pre_data):] + 1
            # save the data with the name
            pre_name = name[i] + '_pre_k=' + str(k) + '.csv'
            post_name = name[i] + '_post_k=' + str(k) + '.csv'
            pre_data = add_label(pre_data, k)
            post_data = add_label(post_data, k)
            # the first half is pre, the second half is post
            csv_pre = pd.DataFrame(pre_data, columns=metrics_names[k-2])
            csv_post = pd.DataFrame(post_data, columns=metrics_names[k-2])
            # delimitors are ;
            csv_pre.to_csv(os.path.join(folder_path, pre_name), sep=';', index=False)
            csv_post.to_csv(os.path.join(folder_path, post_name), sep=';', index=False)
    # save the centroids
    centroids = pd.DataFrame(centroids, columns=['name', 'centroids'])
    centroids.to_csv(os.path.join(folder_path, 'centroids.csv'), sep=';', index=False)

# 2. group by all subjects
def group_by_all(pre, post, name, metrics_names=metrics_names):
    # create a folder to save the data named group_by_all
    folder_name = 'group_by_all'
    folder_path = os.path.join(VRP_path, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    centroids = []
    
    # kmeans clustering
    for k in range(2, 6):
        # concatenate all the data and remember the index
        pre_data = np.array(pre[0])
        post_data = np.array(post[0])
        index = [len(pre_data)]
        for i in range(1, len(pre)):
            pre_data = np.concatenate((pre_data, np.array(pre[i])), axis=0)
            post_data = np.concatenate((post_data, np.array(post[i])), axis=0)
            index.append(len(pre_data))
        # standardize the data
        data = np.concatenate((pre_data, post_data), axis=0)
        data = data[:, 3:9]
        standard_scaler = StandardScaler()
        standard_scaler.fit(data)
        data = standard_scaler.transform(data)
        # normalize the data
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(data)
        data = minmax_scaler.transform(data)
        # kmeans clustering
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=500, random_state=None)
        labels = kmeans.fit_predict(data)
        labels = reorder_clusters(data, labels)
        # save the centroids
        centroid_name = 'all' + '_k=' + str(k)
        centroid = kmeans.cluster_centers_
        centroid = minmax_scaler.inverse_transform(centroid)
        centroid = standard_scaler.inverse_transform(centroid)
        centroids.append([centroid_name, centroid])
        pre_labels = labels[:len(pre_data)]
        post_labels = labels[len(pre_data):]
        # match the original index for pre and post, index should be divided into two parts
        for i in range(len(index)):
            if i == 0:
                pre_data[:index[i], 9] = pre_labels[:index[i]] + 1
                csv_pre = pre_data[:index[i]]
            else:
                pre_data[index[i-1]:index[i], 9] = pre_labels[index[i-1]:index[i]] + 1
                csv_pre = pre_data[index[i-1]:index[i]]
            if i == 0:
                post_data[:index[i], 9] = post_labels[:index[i]] + 1
                csv_post = post_data[:index[i]]
            else:
                post_data[index[i-1]:index[i], 9] = post_labels[index[i-1]:index[i]] + 1
                csv_post = post_data[index[i-1]:index[i]]
            # save the data with the name
            pre_name = name[i] + '_pre_k=' + str(k) + '.csv'
            post_name = name[i] + '_post_k=' + str(k) + '.csv'
            csv_pre = add_label(csv_pre, k)
            csv_post = add_label(csv_post, k)
            csv_pre = pd.DataFrame(csv_pre, columns=metrics_names[k-2])
            csv_post = pd.DataFrame(csv_post, columns=metrics_names[k-2])
            # delimitors are ;
            csv_pre.to_csv(os.path.join(folder_path, pre_name), sep=';', index=False)
            csv_post.to_csv(os.path.join(folder_path, post_name), sep=';', index=False)
    # save the centroids
    centroids = pd.DataFrame(centroids, columns=['name', 'centroids'])
    centroids.to_csv(os.path.join(folder_path, 'centroids.csv'), sep=';', index=False)

# 3. group by pres and posts
def group_by_pres_and_posts(pre, post, name, metrics_names=metrics_names):
    # create a folder to save the data named group_by_pres_and_posts
    folder_name = 'group_by_pres_and_posts'
    folder_path = os.path.join(VRP_path, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


    # kmeans clustering
    for k in range(2, 6):
        # concatenate all the data and remember the index
        pre_data = np.array(pre[0])
        post_data = np.array(post[0])
        index = [len(pre_data)]
        for i in range(1, len(pre)):
            pre_data = np.concatenate((pre_data, np.array(pre[i])), axis=0)
            post_data = np.concatenate((post_data, np.array(post[i])), axis=0)
            index.append(len(pre_data))
        data = pre_data[:, 3:9]
        # standardize the data, start with pre
        standard_scaler = StandardScaler()
        standard_scaler.fit(data)
        data = standard_scaler.transform(data)
        # normalize the data
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(data)
        data = minmax_scaler.transform(data)
        pre_centroids = []
        # kmeans clustering
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=500, random_state=None)
        labels = kmeans.fit_predict(data)
        labels = reorder_clusters(data, labels)
        #   save the centroids
        centroid_name = 'pre' + '_k=' + str(k)
        centroid = kmeans.cluster_centers_
        centroid = minmax_scaler.inverse_transform(centroid)
        centroid = standard_scaler.inverse_transform(centroid)
        pre_centroids.append([centroid_name, centroid])
        # match the original index
        for i in range(len(index)):
            if i == 0:
                pre_data[:index[i], 9] = labels[:index[i]] + 1
                csv_pre = pre_data[:index[i]]
            else:
                pre_data[index[i-1]:index[i], 9] = labels[index[i-1]:index[i]] + 1
                csv_pre = pre_data[index[i-1]:index[i]]
            csv_pre = add_label(csv_pre, k)
            csv_pre = pd.DataFrame(csv_pre, columns=metrics_names[k-2])
        # save the data with the name
            pre_name = name[i] + '_pre_k=' + str(k) + '.csv'
            # delimitors are ;
            csv_pre.to_csv(os.path.join(folder_path, pre_name), sep=';', index=False)
    # save the centroids
    pre_centroids = pd.DataFrame(pre_centroids, columns=['name', 'centroids'])
    pre_centroids.to_csv(os.path.join(folder_path, 'pre_centroids.csv'), sep=';', index=False)
    
    # kmeans clustering
    for k in range(2, 6):
        # then post
        # concatenate all the data and remember the index
        pre_data = np.array(pre[0])
        post_data = np.array(post[0])
        index = [len(pre_data)]
        for i in range(1, len(pre)):
            pre_data = np.concatenate((pre_data, np.array(pre[i])), axis=0)
            post_data = np.concatenate((post_data, np.array(post[i])), axis=0)
            index.append(len(pre_data))
        data = post_data[:, 3:9]
        standard_scaler = StandardScaler()
        standard_scaler.fit(data)
        data = standard_scaler.transform(data)
        # normalize the data
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(data)
        data = minmax_scaler.transform(data)
        post_centroids = []
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=500, random_state=None)
        labels = kmeans.fit_predict(data)
        labels = reorder_clusters(data, labels)
        #save the centroids
        centroid_name = 'post' + '_k=' + str(k)
        centroid = kmeans.cluster_centers_
        centroid = minmax_scaler.inverse_transform(centroid)
        centroid = standard_scaler.inverse_transform(centroid)
        post_centroids.append([centroid_name, centroid])

        # match the original index
        for i in range(len(index)):
            if i == 0:
                post_data[:index[i], 9] = labels[:index[i]] + 1
                csv_post = post_data[:index[i]]
            else:
                post_data[index[i-1]:index[i], 9] = labels[index[i-1]:index[i]] + 1
                csv_post = post_data[index[i-1]:index[i]]
            
            csv_post = add_label(csv_post, k)
            csv_post = pd.DataFrame(csv_post, columns=metrics_names[k-2])
        # save the data with the name
            post_name = name[i] + '_post_k=' + str(k) + '.csv'
            # delimitors are ;
            csv_post.to_csv(os.path.join(folder_path, post_name), sep=';', index=False)
    # save the centroids
    post_centroids = pd.DataFrame(post_centroids, columns=['name', 'centroids'])
    post_centroids.to_csv(os.path.join(folder_path, 'post_centroids.csv'), sep=';', index=False)

# 4.group by only the CSE and CPPs
def group_by_CSE_CPPs(pre, post, name, metrics_names=metrics_names):
    # create a folder to save the data named group_by_subject
    folder_name = 'group_by_CSE_CPPs'
    folder_path = os.path.join(VRP_path, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    centroids = []
    # each clustering only focuses on one subject, so normalize and normalize pre and post data by subject
    for i in range(len(pre)):
        
        # kmeans clustering
        for k in range(2, 6):
            pre_data = np.array(pre[i])
            post_data = np.array(post[i])
            # concatenate the pre and post data for each subject, only CSE and CPPs
            data = np.concatenate((pre_data[:, 5:6], post_data[:, 5:6]), axis=0)
            # standardize the data
            standard_scaler = StandardScaler()
            standard_scaler.fit(data)
            data = standard_scaler.transform(data)
            # normalize the data
            minmax_scaler = MinMaxScaler()
            minmax_scaler.fit(data)
            data = minmax_scaler.transform(data)
            # kmeans clustering
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=500, random_state=None)
            labels = kmeans.fit_predict(data)
            labels = reorder_clusters(np.concatenate((pre_data, post_data), axis=0), labels)
            #save the centroids
            centroid_name = name[i] + '_k=' + str(k)
            centroid = kmeans.cluster_centers_
            centroid = minmax_scaler.inverse_transform(centroid)
            centroid = standard_scaler.inverse_transform(centroid)
            centroids.append([centroid_name, centroid])
            # match the original index
            pre_data[:, 9] = labels[:len(pre_data)] + 1
            post_data[:, 9] = labels[len(pre_data):] + 1
            # save the data with the name
            pre_name = name[i] + '_pre_k=' + str(k) + '.csv'
            post_name = name[i] + '_post_k=' + str(k) + '.csv'
            pre_data = add_label(pre_data, k)
            post_data = add_label(post_data, k)
            # the first half is pre, the second half is post
            csv_pre = pd.DataFrame(pre_data, columns=metrics_names[k-2])
            csv_post = pd.DataFrame(post_data, columns=metrics_names[k-2])
            # delimitors are ;
            csv_pre.to_csv(os.path.join(folder_path, pre_name), sep=';', index=False)
            csv_post.to_csv(os.path.join(folder_path, post_name), sep=';', index=False)
    # save the centroids
    centroids = pd.DataFrame(centroids, columns=['name', 'centroids'])
    centroids.to_csv(os.path.join(folder_path, 'centroids.csv'), sep=';', index=False)

# 5. group by surgery type, only total and partial.
'''
This function is unfinished. Modify it to directly process the VRP file and then use the 123 function to generate.
'''
def group_by_sur_type(patient_pre_data, patient_post_data, criteria, list_of_patients, metrics_names=metrics_names):            
    for i in range(len(criteria)):
        
        # create a folder to save the data named by list_of_patients[i]
        folder_name = 'all'
        subfoler_name = criteria[i]
        folder_path = os.path.join(VRP_path, folder_name, subfoler_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        centroids = []
        pre = patient_pre_data[i]
        post = patient_post_data[i]
        for i in range(len(pre)):
            for j in range(len(pre[i])):
                pre[i][j][7] = np.log10(pre[i][j][7])
                post[i][j][7] = np.log10(post[i][j][7])
        # same as group_by_subject
        for i in range(len(pre)):
            # kmeans clustering
            for k in range(2, 6):
                pre_data = np.array(pre[i])
                post_data = np.array(post[i])
                # concatenate the pre and post data for each subject
                data = np.concatenate((pre_data[:, 3:9], post_data[:, 3:9]), axis=0)
                # standardize the data
                standard_scaler = StandardScaler()
                standard_scaler.fit(data)
                data = standard_scaler.transform(data)
                # normalize the data
                minmax_scaler = MinMaxScaler()
                minmax_scaler.fit(data)
                data = minmax_scaler.transform(data)
                # kmeans clustering
                kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=500, random_state=None)
                labels = kmeans.fit_predict(data)
                labels = reorder_clusters(np.concatenate((pre_data, post_data), axis=0), labels)
                #save the centroids
                centroid_name = name[i] + '_k=' + str(k)
                centroid = kmeans.cluster_centers_
                centroid = minmax_scaler.inverse_transform(centroid)
                centroid = standard_scaler.inverse_transform(centroid)
                centroids.append([centroid_name, centroid])
                # match the original index
                pre_data[:, 9] = labels[:len(pre_data)] + 1
                post_data[:, 9] = labels[len(pre_data):] + 1
                # save the data with the name
                pre_name = name[i] + '_pre_k=' + str(k) + '.csv'
                post_name = name[i] + '_post_k=' + str(k) + '.csv'
                pre_data = add_label(pre_data, k)
                post_data = add_label(post_data, k)
                # the first half is pre, the second half is post
                csv_pre = pd.DataFrame(pre_data, columns=metrics_names[k-2])
                csv_post = pd.DataFrame(post_data, columns=metrics_names[k-2])
                # delimitors are ;
                csv_pre.to_csv(os.path.join(folder_path, pre_name), sep=';', index=False)
                csv_post.to_csv(os.path.join(folder_path, post_name), sep=';', index=False)
        # save the centroids
        centroids = pd.DataFrame(centroids, columns=['name', 'centroids'])
        centroids.to_csv(os.path.join(folder_path, 'centroids.csv'), sep=';', index=False)
    
# 6. plot BIC curve for each subject
def BIC_curve(pre, post, name):
    # 1. group by each subject, subject count X k centroids
    # create a folder to save the data named group_by_subject
    folder_name = 'BIC'
    folder_path = os.path.join(VRP_path, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    centroids = []
    # each clustering only focuses on one subject, so normalize and normalize pre and post data by subject
    for i in range(len(pre)):
        bics = []
        # kmeans clustering
        for k in range(1, 10):
            pre_data = np.array(pre[i])
            post_data = np.array(post[i])
            # concatenate the pre and post data for each subject
            data = np.concatenate((pre_data[:, 3:9], post_data[:, 3:9]), axis=0)
            # standardize the data
            standard_scaler = StandardScaler()
            standard_scaler.fit(data)
            data = standard_scaler.transform(data)
            # normalize the data
            minmax_scaler = MinMaxScaler()
            minmax_scaler.fit(data)
            data = minmax_scaler.transform(data)
            # kmeans clustering
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=500, random_state=None)
            labels = kmeans.fit_predict(data)
            labels = reorder_clusters(np.concatenate((pre_data, post_data), axis=0), labels)
            # get the BIC
            bic = compute_bic(kmeans, data)
            bics.append(bic)
        plt.plot(range(1, 10), bics, marker='o')
        # title named by name[i]
        plt.title('BIC for patient ' + name[i])
        # set axis
        plt.xlabel('Number of clusters')
        plt.show()
        # then close the plot
        plt.close()

    return


# 7. plot BIC curve for all as a whole
def BIC_curve_all(pre, post, name):
    # kmeans clustering
    bics = []
    for k in range(1, 20):
        # concatenate all the data and remember the index
        pre_data = np.array(pre[0])
        post_data = np.array(post[0])
        index = [len(pre_data)]
        for i in range(1, len(pre)):
            pre_data = np.concatenate((pre_data, np.array(pre[i])), axis=0)
            post_data = np.concatenate((post_data, np.array(post[i])), axis=0)
            index.append(len(pre_data))
        # standardize the data
        data = np.concatenate((pre_data, post_data), axis=0)
        data = data[:, 3:9]
        standard_scaler = StandardScaler()
        standard_scaler.fit(data)
        data = standard_scaler.transform(data)
        # normalize the data
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(data)
        data = minmax_scaler.transform(data)
        # kmeans clustering
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=500, random_state=None)
        labels = kmeans.fit_predict(data)
        labels = reorder_clusters(data, labels)
        # get the BIC
        bic = compute_bic(kmeans, data)
        bics.append(bic)
    plt.plot(range(1, 20), bics, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('BIC')
    # legend of x axis should be integers
    plt.xticks(np.arange(1, 20, 1))
    # title named by name[i]
    plt.title('BIC for all patients')
    plt.show()
    # then close the plot
    plt.close()


# usage if I want to use it in the future
# filepath = r'L:\Huanchen\Thyrovoice\VRPstats.xlsx'
# criteria = ['Total', 'Partial']  # Add as many criteria as required
# list_of_patients = select_patients(filepath, 'Chirurgie', criteria)
# pre_VRP_data, post_VRP_data = collect_VRP_data_v3(path, list_of_patients)
# group_by_sur_type(pre_VRP_data, post_VRP_data, criteria, list_of_patients)

    return

def reorder_clusters(original_data, labels):
    # Calculate average for col 1 for each cluster
    cluster_averages = {}
    for label in set(labels):
        cluster_averages[label] = np.mean(original_data[labels == label, 1])
    label_list = labels.copy()
    # sort the cluster averages
    sorted_cluster_averages = sorted(cluster_averages.items(), key=lambda kv: kv[1])

    # reorder the labels
    for i, (label, _) in enumerate(sorted_cluster_averages):
        label_list[labels == label] = i
        # drop the processed label in labels
        
    return label_list

def add_label(data, k):
    # Assuming log_range is a numpy array and k is defined
    rows, cols = data.shape
    data = data[:,:-2]
    # Expand log_range by k columns with zeros
    data = np.hstack([data, np.zeros((rows, k))])

    # Iterate and update values
    for i in range(1, k + 1):
        idx = np.where(data[:, 9] == i)[0]  # Note: MATLAB is 1-indexed, but Python is 0-indexed
        data[idx, 9 + i] = data[idx, 2]  # Same note on indexing

    return data

def select_patients(filepath, colname, criteria=None):
    '''
    load an excel file with the patient info, output the selected patients by matching the values
    :param filepath: Path to the Excel file.
    :param criteria: list of values to select.
    :return: DataFrame of selected patients.
    '''
    # load excel and get the data from sheet 1
    df = pd.read_excel(filepath, sheet_name='main data (2)', engine='openpyxl')
    list_of_patients = []

    # find the col named 'colname'
    for i in range(len(df.columns)):
        if df.columns[i] == colname:
            col_index = i
            break

    # find the patients that match the criteria
    if criteria:
        if len(criteria) == 1:    
            for i in range(len(df)):
                if df.iloc[i, col_index] in criteria:
                    name = df.iloc[i, 0]
                    matching_patients.append(name[:6])
        else:
            for j in range(len(criteria)):
                matching_patients = []
                for i in range(len(df)):
                    if df.iloc[i, col_index] == criteria[j]:
                        name = df.iloc[i, 0]
                        matching_patients.append(name[:6])
                list_of_patients.append(matching_patients)

    return list_of_patients

def compute_bic(kmeans, data):
    n = len(data)
    d = data.shape[1]
    sse = kmeans.inertia_  # sum of squared distances to closest cluster center
    k = kmeans.n_clusters
    bic = sse + k * np.log(n) * d
    return bic



group_by_subject(pre_VRP_data, post_VRP_data, name)
# group_by_all(pre_VRP_data, post_VRP_data, name)
# group_by_pres_and_posts(pre_VRP_data, post_VRP_data, name)
# group_by_CSE_CPPs(pre_VRP_data, post_VRP_data, name)
# BIC_curve_all(pre_VRP_data, post_VRP_data, name)




# group by surgery type
# load the patients with name in selected_patients,take turns to run the group_by_subject function


# group by damage type

# BIC curve, in case study