#!/usr/bin/python
from __future__ import division

import sys
import numpy as np
import pandas as pd
import copy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

#Q1
def loadData(fileDj):
    data = np.loadtxt(fileDj, dtype=float)
    return data

## K-means functions 

def getInitialCentroids(X, k):
    np.random.seed(1)
    initialCentroids_x = np.random.randint(np.min(X[:,0]), np.max(X[:,0]), size=k)
    initialCentroids_y = np.random.randint(np.min(X[:,1]), np.max(X[:,1]), size=k)
    initialCentroids = np.array(list(zip(initialCentroids_x, initialCentroids_y)), dtype=np.float32)
    return initialCentroids

def getDistance(pt1, pt2):
    dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    return dist

def allocatePoints(X, clusters):
    points = pd.DataFrame(X, columns=['X', 'Y'])
    cluster = []
    for i in X:
        minimum = 100
        cluster_val = 0
        for j in range(len(clusters)):
            distance = getDistance(i, clusters[j])
            if distance < minimum:
                minimum = distance
                cluster_val = float(j)
        cluster.append(cluster_val)
    points['Nearest Cluster'] = cluster
    clusters = points
    return clusters

def updateCentroids(clusters, k):
    updatedCentroids = []
    for i in range(0, k):
        x_avg = clusters[clusters['Nearest Cluster'] == i]['X'].mean()
        y_avg = clusters[clusters['Nearest Cluster'] == i]['Y'].mean()
        updatedCentroids.append([x_avg, y_avg])   
    clusters = np.array(updatedCentroids, dtype=np.float32)
    return clusters

#Q2
def kmeans(X, k, maxIter=1000):
    X = X[:, 0:2]
    clusters = getInitialCentroids(X, k)
    for i in range(0, maxIter):
        check = copy.deepcopy(clusters)
        clusters = allocatePoints(X, clusters)
        clusters = updateCentroids(clusters, k)
        if np.array_equal(check, clusters) == True:
            points = allocatePoints(X, clusters)
            break
    return clusters, points

def scikitKmeans(X, k):
    X = X[:, 0:2]
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    return centroids

#Q3
def visualizeClusters(points, clusters):
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    plt.figure(1)
    for i in range(len(clusters)):
        points_i = points[points['Nearest Cluster'] == i]
        plt.scatter(points_i['X'].values, points_i['Y'].values, c=colors[i], s=7)
        plt.scatter(clusters[:,0], clusters[:,-1], marker='*', s=300, c='black')
    return 0

#Q4
def kneeFinding(X, kList):
    objective_function = []
    for i in kList:
        clusters_k = kmeans(X, i, maxIter=1000)
        
        objective_function.append(sum(np.min(cdist(X[:, 0:2], clusters_k[0], 'euclidean'), axis=1)) / X.shape[0])
        #print(objective_function)
    plt.figure(2)
    plt.plot(kList, objective_function, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Objective Function')
    plt.title('k vs. Objective Function')
    plt.show()
    return 0

def purity(predicted, trueLabels, k):
    purities = []
    l = 1.0
    for p in range(0, k): 
        predicted = predicted.replace(float(k)-l,float(k)-p)
        l+=1
    predicted['True Cluster'] = trueLabels[:,-1]
    for i in range(1, k+1):
        predicted_i = predicted[predicted['Nearest Cluster'] == i]
        cluster_len = len(predicted_i)
        max_ = 0
        for j in range(1, k+1):
            predicted_j = len(predicted_i[predicted_i['True Cluster'] == j])
            if predicted_j > max_:
                max_ = predicted_j 
        purities.append((1/cluster_len)*max_)
    return purities


def main():
    #######dataset path
    datadir = sys.argv[1] 
    datadir = 'data_sets_clustering/'
    pathDataset1 = datadir+'/humanData.txt'
    #Q1
    dataset1 = loadData(pathDataset1)
    #pathDataset2 = datadir+'/audioData.txt'
    #dataset2 = loadData(pathDataset2)
    
    #Q2 and Q3
    clusters, points = kmeans(dataset1, 2, maxIter=1000)
    print(clusters)
    visualizeClusters(points, clusters)
 
    #Q4
    kneeFinding(dataset1,range(1,7))
    
    #Q5
    purities = purity(points, dataset1, 2)
    print(purities)

if __name__ == "__main__":
    main()