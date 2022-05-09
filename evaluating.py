### 4.EVALUATING THE CLUSTERING METHOD

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

from kneed import DataGenerator, KneeLocator

import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score

def normalizer(df):
    feature_vector = df.to_numpy()
    # Create a scaler object
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    # Apply scaler object to normalize feature vectors between 0 and 1
    features = scaler.fit_transform(feature_vector)
    return features

if __name__ == "__main__":
    
    print('Evaluating the model...')
    df = pd.read_csv('dataset.csv')
    df = df.dropna() # Remove NaN values
    # Reset the dataframe index
    df = df.reset_index(drop=True)
    
    # Set the selected feature vectors
    df_variance = df[['SRm', 'SBm', 'MFCC4', 'SCm', 'ZCRv', 'MFCC6', 'SRv', 'MFCC7', 'MFCC5','SFm', 'duration', 'ZCRm', 'MFCC8']]
    df_laplacian = df[['SBm', 'SRm', 'MFCC4', 'SCm', 'MFCC5', 'MFCC7', 'SFMm', 'MFCC3','MFCC6', 'MFCC2', 'ZCRm', 'ZCRv', 'ENm']]

    # Normalize the feature vectors
    variance = normalizer(df_variance)
    laplacian = normalizer(df_laplacian)

    # Apply non-metric dimensionality reduction algorithm with t-SNE, to project the data in two dimensions
    tsne = sklearn.manifold.TSNE(n_components=2, perplexity=45, init='pca', n_iter=1500, n_iter_without_progress=200, learning_rate='auto')
    X_tnse_var = tsne.fit_transform(variance)
    X_tnse_lap = tsne.fit_transform(laplacian)

    # Find the point of maximum curvature        
    n_neighbors = 13*2 # Number of dimensions * 2
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)

    neighbors_var = neighbors.fit(X_tnse_var)
    distances_var, indices_var = neighbors_var.kneighbors(X_tnse_var)
    distances_var = np.sort(distances_var, axis=0)
    distances_var = distances_var[:,n_neighbors-1]
    i = np.arange(len(distances_var))
    knee_var = KneeLocator(i, distances_var, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    eps_var = distances_var[knee_var.knee]

    neighbors_lap = neighbors.fit(X_tnse_lap)
    distances_lap, indices_lap = neighbors_lap.kneighbors(X_tnse_lap)
    distances_lap = np.sort(distances_lap, axis=0)
    distances_lap = distances_lap[:,n_neighbors-1]
    dist = np.arange(len(distances_lap))
    knee_lap = KneeLocator(dist, distances_lap, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    eps_lap = distances_lap[knee_lap.knee]
    
    n_samples_var = []
    sihlouette_var = []
    n_samples_lap = []
    sihlouette_lap = []
    
    print('Finding optimal parameters for the model...')
    for samples in tqdm([20,25,30,35,40,45,50,55,60,65,70,75]):
        dbscan_var = sklearn.cluster.DBSCAN(eps=eps_var, min_samples=samples)
        label_var = dbscan_var.fit_predict(X_tnse_var)
        dbscan_lap = sklearn.cluster.DBSCAN(eps=eps_lap, min_samples=samples)
        label_lap = dbscan_lap.fit_predict(X_tnse_lap)

        noise_var_index = (dbscan_var.labels_ == -1)
        silhouette_per_sample_var = silhouette_samples(X_tnse_var, dbscan_var.labels_, metric='euclidean')
        silhouette_of_non_noise_var = silhouette_per_sample_var[~noise_var_index]
        n_samples_var.append(samples)
        sihlouette_var.append(silhouette_of_non_noise_var.mean())

        noise_lap_index = (dbscan_lap.labels_ == -1)
        silhouette_per_sample_lap = silhouette_samples(X_tnse_lap, dbscan_lap.labels_, metric='euclidean')
        silhouette_of_non_noise_lap = silhouette_per_sample_lap[~noise_lap_index]
        n_samples_lap.append(samples)
        sihlouette_lap.append(silhouette_of_non_noise_lap.mean())

    # Plot the results
    fig, ax = plt.subplots(figsize=(10,10))
    plt.plot(n_samples_var, sihlouette_var, linewidth=3, label='Variance')
    plt.plot(n_samples_lap, sihlouette_lap, linewidth=3, label='Laplacian')
    plt.xlabel('Minimum number of samples', fontsize=25)
    plt.ylabel('Silhouette scores', fontsize=25)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    plt.savefig('silhouette.jpg')
    
    # Find the optimal value for min_samples
    if max(sihlouette_var) > max(sihlouette_lap):
        idx = np.argsort(sihlouette_var)[-1] # maximum silhouette score index
        cluster = sklearn.cluster.DBSCAN(eps=eps_var, min_samples=n_samples_var[idx]).fit(X_tnse_var)
        X_tnse = X_tnse_var 
        selection = 'Variance'
    else:
        idx = np.argsort(sihlouette_lap)[-1] # maximum silhouette score index
        cluster = sklearn.cluster.DBSCAN(eps=eps_lap, min_samples=n_samples_lap[idx]).fit(X_tnse_lap)
        X_tnse = X_tnse_lap 
        selection = 'Laplacian'

    print(f"Number of bird syllables clusters found: {np.unique(cluster.labels_).size-1}")
    print(f"Number of noisy samples {len(cluster.labels_[cluster.labels_==-1])} out of a total of {len(cluster.labels_)} samples")
    # Compute scores (output is in range 0 to 1 where 1 is best)
    noise_features_index = (cluster.labels_ == -1)
    silhouette_per_sample = silhouette_samples(X_tnse, cluster.labels_, metric='euclidean')
    silhouette_of_non_noise_samples = silhouette_per_sample[~noise_features_index]
    print(f'\nSilhouette Coefficient score: {silhouette_of_non_noise_samples.mean()}')
       
    for lab_val in np.unique(cluster.labels_):

        if lab_val == -1: # displaying noisy samples in white
            ax.scatter(X_tnse[cluster.labels_==lab_val, 0], 
                       X_tnse[cluster.labels_==lab_val, 1], c='gray', alpha=0.1, label='Noise')
            ax.legend()
        else:
            rgb = (random.random(), random.random(), random.random())
            ax.scatter(X_tnse[cluster.labels_==lab_val, 0], X_tnse[cluster.labels_==lab_val, 1], c=[rgb], alpha=0.8)

    ax.set_title(f'DBSCAN clustering using {selection} vector', fontsize=25)
    ax.set_xlabel('t-SNE component 1', fontsize=20)
    ax.set_ylabel('t-SNE component 2', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.show()
    plt.savefig('dbscan.jpg')
        
    # Plot the bird syllables spectrograms
    zero = np.asarray(np.where(cluster.labels_ == 0)).flatten()[10:17]
    one = np.asarray(np.where(cluster.labels_ == 1)).flatten()[10:17]
    indexes = np.append(zero, one)
    two = np.asarray(np.where(cluster.labels_ == 2)).flatten()[10:17]
    indexes = np.append(indexes, two)
    three = np.asarray(np.where(cluster.labels_ == 3)).flatten()[10:17]
    indexes = np.append(indexes, three)

    fig, axes = plt.subplots(4,7, subplot_kw={'xticks':(), 'yticks':()}, figsize=(10,10))
    for i, ax in zip(indexes, axes.ravel()):
        y, sr = librosa.load(df['file-name'][i], sr=22050)
        S = np.abs(librosa.stft(y[df.start[i]:df.end[i]], n_fft=128))
        ax.imshow(librosa.amplitude_to_db(S, ref=np.max), cmap=plt.cm.binary, aspect='auto', norm=None, vmax=0, vmin=-80)
        ax.invert_yaxis()
    plt.savefig('repertoire.jpg')