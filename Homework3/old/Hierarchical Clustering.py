# Hierarchical Clustering - Decision Making with Matrices (People-Restaurant Matrices)

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

#########################
# Hierarchical Clustering
#########################

linked = linkage(M_restaurant_PCA, 'single')
labelList = list(restaurants)
# explicit interface
fig = plt.figure(figsize = (20,15))
ax = fig.add_subplot(1,1,1)
dendrogram(linked, orientation = 'top', labels = labelList, distanec_sort = 'descending', show_leaf_counts = True, ax = ax)
ax.tick_params(axis = 'x', which = 'major', labelsize = 15)
ax.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax.set_title("Dendrogram Restaurant Hierarchical Clustering Using PCA components", size = 20)
fig.savefig('Dendrogram_Restaurant_PCA.png')
plt.show()

####################
# Clustering Metrics
####################
# https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
print("The Calinski-Harabaz Index is used to measure better defined clusters.")
print("\nThe Calinski-Harabaz score is higher when clusters are dense and well separated.\n")

range_n_clusters = [2,3,4,5,6]
for n_clusters in range_n_clusters:
    ClusterInit = KMeans(n_clusters = n_clusters, random_state = 7)
    cluster_labels = ClusterInit.fit_predict(M_people_PCA)
    CH_score = metrics.calinski_harabasz_score(M_people_PCA, cluster_labels)
    print("The Calinksi-Harabaz score for: ", n_clusters, " clusters is ", CH_score)


print("The Davies-Bouldin Index is used to measure better defined clusters.")
print("\nThe Davies-Bouldin score is lower when clusters are more separated (e.g. better partitioned).\n")
print("Zero is the lowest possible Davies-Bouldin score.\n")

for n_clusters in range_n_clusters:
    ClusterInit = KMeans(n_clusters = n_clusters, random_state = 7)
    cluster_labels = ClusterInit.fit_predict(M_people_PCA)
    DB_score = metrics.davies_bouldin_score(M_people_PCA, cluster_labels)
    print("The Calinksi-Harabaz score for: ", n_clusters, " clusters is ", DB_score)





