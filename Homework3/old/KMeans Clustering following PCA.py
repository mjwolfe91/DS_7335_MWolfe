# KMeans Clustering following PCA Transformation of Restaurant Data Matrix

# Data
restaurants  = {'flacos':{'distance' : 2,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                        },
              'Joes':{'distance' : 5,
                        'novelty' : 1,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Poke':{'distance' : 4,
                        'novelty' : 2,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },                      
              'Sush-shi':{'distance' : 4,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },
              'Chick Fillet':{'distance' : 3,
                        'novelty' : 2,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                      },
              'Mackie Des':{'distance' : 2,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Michaels':{'distance' : 2,
                        'novelty' : 1,
                        'cost': 1,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                      },
              'Amaze':{'distance' : 3,
                        'novelty' : 5,
                        'cost': 2,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },
              'Kappa':{'distance' : 5,
                        'novelty' : 1,
                        'cost': 2,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Mu':{'distance' : 3,
                        'novelty' : 1,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      }                      
                }

# Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
import matplotlib.cm as cm  #https://matplotlib.org/api/cm_api.html
import warnings
warnings.filterwarnings("ignore")

## Restaurant Matrix (M_restaurant)
restaurantsKeys, restaurantsValues = [], [] # Convert each restraunt's values into a list
for k1, v1 in restaurants.items():
    for k2, v2 in v1.items():
        restaurantsKeys.append(k1+'_'+k2)
        restaurantsValues.append(v2)
#print(restaurantsKeys)
#print(restaurantsValues)
# Given that restrauntValues is a list with length 40, it must be converted into array of 10 rows and 4 columns to accurately represent the restraunt data structure
M_restaurant = np.reshape(restaurantsValues, (10,4))

pcaRestaurant = PCA(n_components = 2)
M_restaurant_PCA = pcaRestaurant.fit_transform(M_restaurant)


##########################################
# Principal Components [Restaurant Matrix]
##########################################

# PCA Example - code taken from https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
pcaRestaurant = PCA(n_components = 2)
M_restaurant_PCA = pcaRestaurant.fit_transform(M_restaurant)
print('\n________________________________________________________________')
print('#####    Reducing Restaurant Matrix Dimensionality with PCA  #####')
print('PCA Restaurant Matrix component weights: ', pcaRestaurant.components_)
print('PCA Restaurant Matrix % explained variance: ', pcaRestaurant.explained_variance_)

# Plot Principal Components
fig, ax = plt.subplots(1, 1, figsize=(12,12))
fig.subplots_adjust(left = 0.0625, right = 0.95, wspace = 0.1)
ax.scatter(M_restaurant_PCA[:,0], M_restaurant_PCA[:,1], alpha = 0.2)
draw_vector([0,0], [0,3], ax = ax)
draw_vector([0,0], [3,0], ax = ax)
ax.axis('equal')
ax.set(xlabel = 'Component 1', ylabel = 'Component 2', title = 'Principal Components (Restaurant Matrix)', xlim = (-4,4), ylim = (-4,4))
fig.savefig('PCA_Restaurant_Plot.png')
fig.show()

#################################################################
# Cluster and Plot using PCA transformed data [Restaurant Matrix]
#################################################################
Rkmeans = KMeans(n_clusters = 3)
Rkmeans.fit(M_restaurant_PCA)
R_centroid = Rkmeans.cluster_centers_
R_labels = Rkmeans.labels_
print('\n______________________________________')
print('#####    KMeans Clusters (3) of PCA Transformed Restaurant Matrix centroids:    #####\n')
print(R_centroid)
print('PCA Transformed Restaurant Matrix Labels: ', R_labels)

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

#https://matplotlib.org/users/colors.html
#cluster 0 is green, cluster 1 is red, cluster 2 is cyan (blue)
colors = ["g.","r.","c."]
labelList = ['Flacos', 'Joes', 'Poke', 'Sush-shi', 'Chick Fillet', 'Mackie Des', 'Michaels', 'Amaze', 'Kappa', 'Mu'] #list(restaurants)

for i in range(len(M_restaurant_PCA)):
   print ("coordinate:" , M_restaurant_PCA[i], "label:", R_labels[i])
   ax.plot(M_restaurant_PCA[i][0], M_restaurant_PCA[i][1], colors[R_labels[i]], markersize=10)
   #https://matplotlib.org/users/annotations_intro.html
   #https://matplotlib.org/users/text_intro.html
   ax.annotate(labelList[i], (M_restaurant_PCA[i][0],M_restaurant_PCA[i][1]), size=25)
ax.scatter(R_centroid[:,0],R_centroid[:,1], marker = "x", s=150, c = ['green','red','cyan'], linewidths = 5, zorder =10)
ax.set_title("KMeans Restaurant Clustering using PCA Components", size = 20)
fig.savefig('KMeans_3xClustering_RestaurantPCA.png')
fig.show()