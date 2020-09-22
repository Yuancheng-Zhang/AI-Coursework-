from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

"""Q3: Clustering
Question: Using these values, cluster the source IP addresses by the number of records they
appear in. Repeat for destination IP addresses. Is there an obvious number of clusters 
(and hence, an obvious split in the tally counts)? You should explore using different clustering algorithms and different tools for determining the number of clusters. You might find some of the content of worksheet 1 useful in this context.
"""


"Read the file: "
data = pd.read_csv('coursework1.csv')

"Data Pre-processing"
sourceip = data['sourceIP'].tolist()
destip = data['destIP'].tolist()

"Convert the csv into list and re-order the sequence"
d = pd.Series(destip).value_counts().reset_index().values.tolist()
s = pd.Series(sourceip).value_counts().reset_index().values.tolist()

###############################################################################

def KMPlot(IPList, cluster):
    
    "KMeans Clustering"
    global a
    a= np.zeros(shape=[len(IPList), 2])

    store1 = []
    store2 = []

    K = range(0, len(a))
    for k in K:
        store1.append(IPList[k][0])
        store2.append(IPList[k][1])

    store3 = np.asarray(store2)

    for k in K:
        a[k, 0] = k
        a[k, 1] = store3[k]

    # # Original Figure
    # plt.figure()
    # plt.scatter(a[:, 0], a[:, 1])

    "Plot Kmeans Clustering Figure"
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(a)
    plt.figure()
    plt.scatter(a[:, 0], a[:, 1], c=kmeans.labels_, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
    plt.show()

    return


def ECPlot():
    
    "Elbow Curve"
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(a)
        distortions.append(sum(np.min(cdist(a, kmeanModel.cluster_centers_,
                                            'euclidean') ** 2, axis=1)) / a.shape[0])

    plt.figure()
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Value of k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method determining the optimal k value')
    plt.show()

    return


def HCPlot(fc):
    
    "Hierarchical Clustering"
    linked = linkage(a, 'average')
    labelList = range(0, len(a))
    
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=labelList)
    plt.show()
    
    "fcluster"
    HF = fcluster(linked, fc, criterion='maxclust')
    plt.figure()
    plt.scatter(range(len(a)), a[:,1], c=HF, cmap='rainbow')
    plt.show()

    return


def GMMPlot(cluster):
    
    "GMM"
    gmm = GMM(n_components=cluster).fit(a)
    labels = gmm.predict(a)
    plt.scatter(a[:, 0], a[:, 1], c=labels, s=40, cmap='viridis')
    plt.show()

    return

def SilhKM(X):
    
    "Silhouette for Kmeans"
    silhouette_store = []
    
    # Set the range of n_clusters:
    range_n_clusters = [2, 3, 4, 5, 6]
    K = range(1, 6)

    for n_clusters in range_n_clusters:

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        # print("For n_clusters =", n_clusters,
        #       "The average silhouette_score is :", silhouette_avg)
        
        silhouette_store.append(silhouette_avg)
        M = np.array(silhouette_store)
        
    return silhouette_store, range_n_clusters

def Silh_charts(sil):
    
    "Plot Silhouette Score Chart"
    score, n_clusters = sil
    plt.plot(n_clusters, score)
    plt.ylabel('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.show()
    
    return


#%%
###############################################################################

"Plot 6 Figures of SourceIP"
KMPlot(s, 3)
ECPlot()
HCPlot(4)
GMMPlot(4)
SilhKM(a)
Silh_charts(SilhKM(a))

"Plot 6 Figures of DestIP"
KMPlot(d, 2)
ECPlot()
HCPlot(2)
GMMPlot(2)
SilhKM(a)
Silh_charts(SilhKM(a))
