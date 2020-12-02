import os
import pickle
import keras
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot as plt

from layers import CoAttentionParallel
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy import cluster
from kneed import KneeLocator

# =========================
# load model and data
# =========================

def save_pkl_data(path, data):
    with open(path, 'wb') as fo:
        pickle.dump(data, fo)

def load_pkl_data(path):
    with open(path, 'rb') as fi:
        data = pickle.load(fi)
    return data

# ==============================
# TNSE & hierachical implementation
# ==============================
def plot_with_dbscan_labels(tsne_results, labels, unique_grp, plt_label_2):
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        palette=sns.color_palette("hls", unique_grp),
        hue = labels,
        legend="full",
        alpha=0.3 
    )

    fig_name = "tsne" + plt_label_2 + ".png"
    plt.savefig(fig_name)
    plt.clf()


def apply_tsne(input_data, permission_as_num, unique_grp, label=''):
    print("applying tsne...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(input_data)

    sns_plot = sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        palette=sns.color_palette("hls", unique_grp),
        hue = permission_as_num,
        legend="full",
        alpha=0.3
    )
    fig_name = "tsne" + label + ".png"
    plt.savefig(fig_name)
    plt.clf()

    return tsne_results

def hierarchical_plot(input_data):
    print("hierachical clustering...")
    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, 'hierarchical.dendrogram.pkl')

    plt.title("Dendrograms") 
    Z = shc.linkage(input_data, method='ward') #Linkage matrix
    dend = load_pkl_data(path_data)
    '''
    dend = shc.dendrogram(Z, color_threshold=42)
    plt.savefig('hierarchical_plot_42_clusters.png')
    plt.clf()
    '''
    cutree = cluster.hierarchy.cut_tree(Z,height=42)
    return cutree

def seperate_clusters(cluster_labels, input_data, permission_labels):
    cluster_dict = {} #each cluster label has list of tuple - (point, permission)
    for i in range(len(cluster_labels)):
        cluster_num = cluster_labels[i][0]
        data_point = input_data[i]
        perm = permission_labels[i]

        if cluster_num in cluster_dict:
            cluster_dict[cluster_num].append((data_point, perm))
        else:
            cluster_dict[cluster_num] = [(data_point, perm)]
           
    count_diagrams = 1
    
    for key in cluster_dict.keys():
        data = cluster_dict[key]
        input_data_list, permission_as_num = [], []
        unique_grp = 0
        for i in range(len(data)):
            input_data_list.append(data[i][0])
            if data[i][1] not in permission_as_num:
               unique_grp += 1
            permission_as_num.append(data[i][1])
        
        # plt_label_0 = "_cluster_" + str(count_diagrams)
        # eps_value = compute_eps(input_data_list, plt_label_0)

        plt_label_1 = "_cluster_" + str(count_diagrams)
        tsne_results = apply_tsne(input_data_list, permission_as_num, unique_grp, plt_label_1)

        labels = apply_clustering(input_data_list, str(count_diagrams)) #ndarray structure
        labels = labels.tolist()
        if min(labels) == -1:
            unique_grp = max(labels) + 2
        else:
            unique_grp = max(labels) + 1

        plt_label_2 = "_cluster_" + str(count_diagrams) + "_DBlabels"

        plot_with_dbscan_labels(tsne_results, labels, unique_grp, plt_label_2)

        count_diagrams += 1
    

# =========================
# DBSCAN implementation
# =========================
def compute_eps(input_data, diag_label = ''):
    print("computing eps param")
    fig_name = "eps_value" + diag_label + ".png"

    neigh = NearestNeighbors(n_neighbors=5)
    nbrs = neigh.fit(input_data)
    distances, indices = nbrs.kneighbors(input_data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,4]

    i = np.arange(len(distances))
    kneedle = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    '''
    kneedle.plot_knee()
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.savefig(fig_name)
    plt.clf()
    '''
    return round(kneedle.knee_y, 1)

def apply_clustering(input_data, diag_label = ''):
    print("applying dbscan...")
    eps_len = compute_eps(input_data)
    print(eps_len)
    db = DBSCAN(eps=eps_len, min_samples=5).fit(input_data)
    labels = db.labels_

    return labels

def get_permission_list(permissions):
    permission_list = []
    for permission_set in permissions:
        permission_list.append(list(permission_set))

    return permission_list

def tranform_permission(permission_list):
    perm_as_num= []
    unique_grp = 0
    perm_id = {'NETWORK':1, 'LOCATION':2, 'MICROPHONE':4,'SMS':8, 'CAMERA':16, 'CALL':32, 'STORAGE':64, 'CONTACTS':128}
    for perm in permission_list:
        num = 0
        for p in perm:
            num += perm_id[p]
        if num not in perm_as_num:
            unique_grp += 1
        perm_as_num.append(num)
    
    return perm_as_num, unique_grp

def prepare_apks():
    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, '..', 'data', 'total')

    path_data = os.path.join(path_data, 'outlier.processed.pkl')

    #load our combined pkl file
    #extracted_features shape: (10327,200)
    extracted_features, permissions, outlier_labels, id_labels = load_pkl_data(path_data)
    permission_list = get_permission_list(permissions)
    permission_as_num, unique_grp = tranform_permission(permission_list) #unique group = 44 on whole dataset
    
    cluster_labels = hierarchical_plot(extracted_features) #structure: [array([0]), array([1]),...]
    seperate_clusters(cluster_labels, extracted_features, permission_as_num)
    
def main():
    prepare_apks()


if __name__ == '__main__':
    main()
