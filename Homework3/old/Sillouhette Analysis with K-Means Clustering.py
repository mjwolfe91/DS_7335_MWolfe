# Sillouhette Analysis with K-Means Clustering (3 Clusters) on PCA transformed People Matrix

# Data
people = {'Jane': {'willingness to travel': 0.1596993,
                  'desire for new experience':0.67131344,
                  'cost':0.15006726,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01892,
                  },
          'Bob': {'willingness to travel': 0.63124581,
                  'desire for new experience':0.20269888,
                  'cost':0.01354308,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.15251223,
                  },
          'Mary': {'willingness to travel': 0.49337138 ,
                  'desire for new experience': 0.41879654,
                  'cost': 0.05525843,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.03257365,
                  },
          'Mike': {'willingness to travel': 0.08936756,
                  'desire for new experience': 0.14813813,
                  'cost': 0.43602425,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.32647006,
                  },
          'Alice': {'willingness to travel': 0.05846052,
                  'desire for new experience': 0.6550466,
                  'cost': 0.1020457,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.18444717,
                  },
          'Skip': {'willingness to travel': 0.08534087,
                  'desire for new experience': 0.20286902,
                  'cost': 0.49978215,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.21200796,
                  },
          'Kira': {'willingness to travel': 0.14621567,
                  'desire for new experience': 0.08325185,
                  'cost': 0.59864525,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.17188723,
                  },
          'Moe': {'willingness to travel': 0.05101531,
                  'desire for new experience': 0.03976796,
                  'cost': 0.06372092,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.84549581,
                  },
          'Sara': {'willingness to travel': 0.18780828,
                  'desire for new experience': 0.59094026,
                  'cost': 0.08490399,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.13634747,
                  },
          'Tom': {'willingness to travel': 0.77606127,
                  'desire for new experience': 0.06586204,
                  'cost': 0.14484121,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01323548,
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

##########################
# People Matrix (M_people)
##########################
peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people.items():
    row = []
    
    for k2, v2 in v1.items():
        peopleKeys.append(k1+'_'+k2)
        if k1 == lastKey:
            row.append(v2)      
            lastKey = k1
            
        else:
            peopleValues.append(row)
            row.append(v2)   
            lastKey = k1
#print(peopleKeys)
#print(peopleValues)
M_people = np.array(peopleValues)
M_people.shape

######################################
# Principal Components [People Matrix]
######################################
pcaPeople = PCA(n_components = 2)
M_people_PCA = pcaPeople.fit_transform(M_people)
print('\n____________________________________________________________')
print('#####    Reducing People Matrix Dimensionality with PCA  #####')
print('PCA People Matrix component weights: ', pcaPeople.components_)
print('PCA People Matrix % explained variance: ', pcaPeople.explained_variance_)


# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html 
range_n_clusters = [2,3,4,5,6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches(18,7)

    ######################################
    # First subplot is the Silouhette Plot
    ######################################
    # Silhouette coefficient can range from (-1,1). In this example, all points lie within range [-0.1, 1].
    ax1.set_xlim([-0.1,1])
    ax1.set_ylim([0, len(M_people_PCA) + (n_clusters + 1) * 10])
    # (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters to demarcate clearly

    # Initialize clusters with n_clusters value and a random generator
    ClusterInit = KMeans(n_clusters = n_clusters, random_state = 7)
    cluster_labels = ClusterInit.fit_predict(M_people_PCA)

    # silhouette_score (density and separation of clusters) gives the average value for all the samples
    silhouette_average = metrics.silhouette_score(M_people_PCA, cluster_labels)
    # Compute silhouette_score for each individual sample - score is bound between -1 for incorrect clusters and +1 for highly dense clusters. Scores around 0 indicate overlapping clusters
    silhouette_sampleValues = metrics.silhouette_samples(M_people_PCA, cluster_labels)

    print("\n\n\nFor n_clusters = ", n_clusters,
          "\n\nThe average silhouette_score is: ", silhouette_average,
          "\n\n* The silhouette score is bound between -1 for incorrect clusters and +1 for highly dense clusters.",
          "\n* Scores around 0 indicate overlapping clusters.",
          "\n* The score is higher when clusters are dense and well separated.",
          "\n\nThe individual silhouette scores were: ", silhouette_sampleValues,
          "\n\nAnd their assigned clusters were: ", cluster_labels,
          "\n\nWhich correspond to: 'Jane', 'Bob', 'Mary', 'Mike', 'Alice', 'Skip', 'Kira', 'Moe', 'Sara', 'Tom'")

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i and sort them
        ith_cluster_silhouette_values = \
            silhouette_sampleValues[cluster_labels == i]

        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.rainbow(float(i) / n_clusters)
        ax1.fill_betweenx(np.arrage(y_lower, y_upper),
                          0, ith_cluster_silhouette_values, 
                          facecolor = color, edgecolor = color, alpha = 0.9)

        # Label silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10
    
    ax1.set_title("The silhouette plot for the various clusters.", fontsize = 20)
    ax1.set_xlabel("The silhouette coefficent values", fonsize = 20)
    ax1.set_ylabel("Cluster Label", fontsize = 20)

    # The vertical line for the average silhouette score of all values
    ax1.axvline(x = silhouette_average, color = 'red', linestyle = "--")
    ax1.set_yticks([]) # clear the yaxis labels/ticks
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.xaxis.set_tick_params(labelsize = 20)
    ax1.yaxis.set_tick_params(labelsize = 20)

    ######################################
    # Second subplot shows actual clusters
    ######################################
    colors = cm.rainbow(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(M_people_PCA[:,0], M_people_PCA[:,1], marker = '.', s = 300, lw = 0, alpha = 0.5, c = colors, edgecolor = 'k')
    
    # Label Clusters
    centers = ClusterInit.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:,0], centers[:,1], marker = 'o', c = 'white', alpha = 1, s = 400, edgecolor = 'k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker = '$%d$' % i, alpha = 1, s = 400, edgecolor = 'k')
    ax2.set_title("Visualization of clustered data.", fontsize = 20)
    ax2.set_xlabel("Feature Space of 1st feature", fonsize = 20)
    ax2.set_ylabel("Feature Space of 2nd feature", fontsize = 20)

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters), fontsize = 25, fontweight = 'bold')
    ax2.xaxis.set_tick_params(labelsize = 20)
    ax2.yaxis.set_tick_params(labelsize = 20)
plt.show()
        




# 'Jane', 'Bob', 'Mary', 'Mike', 'Alice', 'Skip', 'Kira', 'Moe', 'Sara', 'Tom'
# Assigned Cluster Groups: [2 1 1 0 2 0 0 0 2 1]

# Group 0: Mike, Skip, Kira, Moe [Favorite = Chick Fillet]
group0 = ranks[0:, [3,5,6,7]]

# Group 1: Bob, Tom [Favorite: Joes]
group1 = ranks[0:, [1,9]]

# Group 3: Jane, Mary, Alice, Sara [Favorite: Amaze]
group2 = ranks[0:, [0,2,4,8]]

# Sums for Each Group: y = 0 (flacos), y = 1 (joes), y = 2 (poke), y = 3 (Sush-shi), y = 4 (Chick Fillet), y = 5 (Mackie Des), y = 6 (Michaels), y = 7 (Amaze), y = 8 (Kappa), y = 9 (Mu)
np.sum(group0, axis = 1) # Group 0 wants to go to Flacos or Chick Fillet (tie)
np.sum(group1, axis = 1) # Group 1 wants to go to Kappas
np.sum(group2, axis = 1) # Group 2 wants to go to Amaze

