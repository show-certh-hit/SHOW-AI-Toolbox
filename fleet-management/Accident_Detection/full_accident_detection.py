import pandas as pd
import numpy as np
import random

# Read the agg_df.csv file
agg_df = pd.read_csv('agg_df_acceleration.csv', index_col=0)
accident_data = pd.read_csv('accident_data_acceleration.csv', index_col=0)

# Remove the index column from accident_data
accident_data = accident_data.iloc[:, 1:]

# Scale the agg_df  
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
agg_df_scaled = scaler.fit_transform(agg_df)
# Normalize the agg_df_scaled
# import the normalize function
import sklearn.preprocessing 
agg_df_normalize = sklearn.preprocessing.normalize(agg_df_scaled)


#Fit and transform the data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.cluster 
import sklearn.metrics
import sklearn.preprocessing


pca = PCA(n_components=None)
pca_components = pca.fit_transform(agg_df_scaled)
plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.xlabel('Principal component')
plt.ylabel('Explained variance ratio')
plt.show()

# Apply the PCA with 3 components
pca = PCA(n_components=2)
agg_df_pca_scaled = pca.fit_transform(agg_df_scaled)
agg_df_pca_normalize = pca.fit_transform(agg_df_normalize)

#Plot the PCA
plt.scatter(agg_df_pca_scaled[:,0], agg_df_pca_scaled[:,1])
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('PCA of the scaled data')
plt.show()

# find the eps and min_samples for DBSCAN model based on silhouette score
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# find the optimal values for eps and min_samples and print the silhouette score
for eps in np.arange(0.1, 2, 0.1):
     for min_samples in range(2, 100, 10):
         db = DBSCAN(eps=eps, min_samples=min_samples)
         db.fit(agg_df_pca_scaled)
         labels = db.labels_
         print('eps: ', eps, 'min_samples: ', min_samples, 'silhouette score: ', silhouette_score(agg_df_pca_scaled, labels))
         # Keep the best values for eps and min_samples
         if silhouette_score(agg_df_pca_scaled, labels) > 0.5:
             print('eps: ', eps, 'min_samples: ', min_samples, 'silhouette score: ', silhouette_score(agg_df_pca_scaled, labels))
             break




# Αpply the DBSCAN model with the best parameters
db = DBSCAN(eps=1, min_samples=2)
db.fit(agg_df_pca_scaled)
labels = db.labels_
# print the silhouette score
print (silhouette_score(agg_df_pca_scaled, labels))
# Print the number of clusters and the number of their points
print (np.unique(labels, return_counts=True))


# Plot the PCA with the clusters
plt.scatter(agg_df_pca_scaled[:,0], agg_df_pca_scaled[:,1], c=labels)
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('PCA of the scaled data')
plt.show()

# Find the silhouette score
silhouette_score(agg_df_pca_scaled, labels)
print (silhouette_score(agg_df_pca_scaled, labels))
# Print the number of clusters and the number of their points
print (np.unique(labels, return_counts=True))

#plot the cluster 0
plt.scatter(agg_df_pca_scaled[db.labels_==0,0], agg_df_pca_scaled[db.labels_==0,1], c='yellow')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('DBSCAN of the scaled data')
plt.show()


# Choose a random point from the accident_data
random_point = random.randint(0, len(agg_df)-1)
random_point = agg_df.iloc[random_point, :]
random_point = random_point.values.reshape(1, -1)
random_point_scaled = scaler.transform(random_point)
random_point_scaled_pca = pca.transform(random_point_scaled)

# Plot the DBSCAN and the accident point
plt.scatter(agg_df_pca_scaled[db.labels_==0,0], agg_df_pca_scaled[db.labels_==0,1], c='yellow')
plt.scatter(random_point_scaled_pca[:,0], random_point_scaled_pca[:,1], c='red')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('DBSCAN of the randow selected scaled data')
plt.show()


# # Choose a random point from the accident_data
random_point = random.randint(0, len(accident_data)-1)
accident_point = accident_data.iloc[random_point, :]
accident_point = accident_point.values.reshape(1, -1)
accident_point_scaled = scaler.transform(accident_point)
accident_point_pca_scaled = pca.transform(accident_point_scaled)

# Plot the DBSCAN and the accident point
plt.scatter(agg_df_pca_scaled[db.labels_==0,0], agg_df_pca_scaled[db.labels_==0,1], c='yellow')
plt.scatter(accident_point_pca_scaled[:,0], accident_point_pca_scaled[:,1], c='red')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('DBSCAN of the accident scaled data')
plt.show()










