# Decision Making with Matricies

# Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from WhatsForLunch import WhatsForLunch

# Sample Data
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

## Rating Scale: 1 is poor and 5 is excellent
restaurants = {'flacos':{'distance' : 2,
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


# Transform data into matrix structures for manipulation
## People Matrix (M_people)
peopleKeys, peopleValues = [], [] # Convert each person's values into a list
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

M_people = np.array(peopleValues) # shape of matrix is (10,4) as expected with 4 attribute values each for 10 people
print('\n____________________________')
print('#####    People Matrix   #####')
print(M_people)

## Restaurant Matrix (M_restaurant)
restaurantsKeys, restaurantsValues = [], []
for k1, v1 in restaurants.items():
    for k2, v2 in v1.items():
        restaurantsKeys.append(k1+'_'+k2)
        restaurantsValues.append(v2)

M_restaurant = np.reshape(restaurantsValues, (10,4))

WhatsForLunch = WhatsForLunch.WhatsForLunch(M_people,M_restaurant)

print('\n_______________________________')
print('#####   Restaurant Matrix   #####')
print(M_restaurant)

''' 1. Informally describe what a linear combination is and how it will relate to our restaurant matrix.'''

print('A linear combination is a sum of products of elements with constant coefficients used to create the products. In our example, a linear combination represents the interaction of person and restaurant to product a weighted score that represents how well that restaurant matches that persons lunch preferences (higher weighted score means a better match).')

''' 2. Choose a person and compute the top restaurant for them using a linear combination.'''

print('According to the linear combinations for Bob, the top restaurant that matches his preference data is Joes, indexed 1. With the highest value with 3.88418002, this matches the restaurants high cost score of 5 and Bobs heavy preference for a cost-effective dining experience with a weight of 0.01354308 (the largest weight associated with the preferences provided). In this linear combination, each entry in the vector represents a weighted value associated with the persons restaurant preferences and how well that restaurant accomodates that preference.')

Bob_lc = WhatsForLunch.get_person_preference(1)
np.argmax(Bob_lc)

print('\n____________________________________________________________________')
print('#####   LC Vector for Bob - All Restaurants    #####')
print(Bob_lc)

''' 3. Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?'''
print('In this resulting matrix multiplication, each i by j matrix entry is the weighted sum of each linear combination of a persons restaurant preferences and the restaurants matched feature with a higher score representing a better match. Each entry in the matrix represents how well a person would enjoy a meal at the restaurant relative to others. Each row represents a person in our matrix and each column a restaurant, and Bob represents the second row of the 10x10 matrix.')

M_people.shape, M_restaurant.shape
M_RxP = WhatsForLunch.match_preference()

print('\n________________________________________________________________________________________')
print('#####   10x10 Restaurant by Person Matrix     #####')
print(M_RxP)

''' 4. Sum all columns in M_PxR matrix to get optimal restaurant for every person. What do the entry's represent?'''
print('The entries in the resulting array are the overall group score for each restraunt based on each persons preferenc. Each entry represents how well each restaurant matches the whole groups lunch preferences relative to one another. The higher the score, the better the restaurant will please the whole group. While it may not be the top choice for anyone, it will be the top choice for everyone.')

RestScores = np.sum(M_RxP, axis = 1)
BestRest_Idx = np.argmax(RestScores)

print('\n_________________________________________________')
print('#####   Best Restaurant - Office     #####')
print('\n       Optimal Restaurant: ', list(restaurants)[BestRest_Idx])

BestRest_PersonIdx = np.argmax(M_RxP, axis = 0)

print('\n_________________________________________________')
print('#####   Best Restaurant for Each Employee     #####')
count = 0
for employee in people.keys():
    print(' Name: ', employee,'     Restaurant: ', list(restaurants)[BestRest_PersonIdx[count]])
    count += 1

''' 5. Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank. Do the same as above to generate the optimal resturant choice.'''

M_RxP_rank = WhatsForLunch.get_best_choice(M_RxP)

np.sum(M_RxP_rank, axis = 1)
np.sum(M_RxP, axis = 1)

''' 6. Why is there a difference between the two?  What problem arrives?  What does represent in the real world?'''

print('When scoring the restaurants originally through linear combinations, the separation between each restaurant was determined by the weighted sum value on a continuous scale. However, this does not apply when ranking because the scale has been changed to discrete values. In this case, we must separate the rankings using computation. As seen when evaluating the rankings with to the scores from the first method, the resulting array is equal across all restaurants. This is not useful for this problem.')
print('In the real world, this method could be a problem because employees without preferences will be forced to follow what other people say because without this information, their opinions will be excluded from the process.')

''' 7. How should you preprocess your data to remove this problem.'''

print('Weighing the ranks on a discrete scale. Principal Components Analysis or K-means clustering methods could also preprocess the data in order to extract sets of employees from the group that have similar preferences and choices.')

''' 8. Find user profiles that are problematic, explain why?'''

print('There is a problem with the scoring of the restaurants that may skew the results of the selection process. If one restaurant gets all 5s in its rating, then that restaurant may come up in group selections more often due to how linear combinations are calculated. Rules limiting how many of certain scores are given out would help control this.')

PeopleScore = np.mean(M_people, axis = 1)
PeopleScore_mean = np.mean(PeopleScore)
ProblemProfile_H = np.where(PeopleScore > PeopleScore_mean*1.25)
ProblemProfile_L = np.where(PeopleScore < PeopleScore_mean*0.8)

print('\n________________________________________________________________________________________')
print('#####    Problematic People Profiles (Variations of more than 25% from mean scores)  #####')
print('\n       Mean People Scores:     ', PeopleScore_mean)
for high_profiles in ProblemProfile_H[0]:
    print('     NAME: ', list(people)[high_profiles],'  HIGH SCORE: ', PeopleScore[high_profiles])
for low_profiles in ProblemProfile_L[0]:
    print('     NAME: ', list(people)[low_profiles],'  HIGH SCORE: ', PeopleScore[low_profiles])


''' 9. Think of two metrics to compute the dissatistifaction with the group.'''

print('We will need to bear in mind there may be clusters of preferences within the group, both in terms of positive and negative preference. For simplicity, assume two distinct preference clusters within the group of 10 employees going out to lunch. Clustering metrics identifying how these two subgroups form can manage dissatisfaction certain employees may have from restaurant heuristics previously measured optimal for the whole group when considering every individuals personal preferences. This is the basis of reward learning.')
print('The Calinski-Harabaz index can be used to measure how well-defined distinct clusters are within the entire group of employees. With this metric, a higher score indicates that the clusters are dense and well-separated. This can create well-understood dissatisfaction clusters for this purpose.')
print('An alternative metric to measure how well-defined distinct clusters are within the entire group is the Davies-Bouldin index. A lower score for this metric indicates more separation between clusters and therefore better partitioned clusters with 0 being the lowest score possible.')
print('Analyzing the clustering metrics previously mentioned, it appears that the preferences between employees are quite different as the metrics imply that from the group of 10, increasing the number of groups better accomodates for individual preferences into distinct group opinions about the best place for lunch. The decision of one optimal lunch spot for the entire group of employees may not accomodate everyones preferences in the most optimal manner. It may be better to manifest the clusters as subgroups according to like preferences.')

##########################################################################
# Plot heatmap - https://seaborn.pydata.org/generated/seaborn.heatmap.html 
##########################################################################
xlist = list(people)
ylist = list(restaurants)
plot_dims = (12,10)
fig, ax = plt.subplots(figsize=plot_dims)
sns.heatmap(ax=ax, data=M_RxP, annot=True, xticklabels=xlist, yticklabels=ylist)
plt.show()

# What is the problems if we want to do clustering with this matrix?
# The problem is that there are too many dimensions in this matrix, we would have to look into techniques to reduce dimensionality so that we could be able to use clustering (i.e. PCA).

######################################
# Principal Components [People Matrix]
######################################

# PCA Example - code taken from https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
pcaPeople = PCA(n_components = 2)
M_people_PCA = pcaPeople.fit_transform(M_people)
print('\n____________________________________________________________')
print('#####    Reducing People Matrix Dimensionality with PCA  #####')
print('PCA People Matrix component weights: ', pcaPeople.components_)
print('PCA People Matrix % explained variance: ', pcaPeople.explained_variance_)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# Plot Principal Components
fig, ax = plt.subplots(1, 1, figsize=(12,12))
fig.subplots_adjust(left = 0.0625, right = 0.95, wspace = 0.1)
ax.scatter(M_people_PCA[:,0], M_people_PCA[:,1], alpha = 0.2)
draw_vector([0,0], [0,1], ax = ax)
draw_vector([0,0], [1,0], ax = ax)
ax.axis('equal')
ax.set(xlabel = 'Component 1', ylabel = 'Component 2', title = 'Principal Components (People Matrix)', xlim = (-4,4), ylim = (-4,4))
fig.savefig('PCA_People_Plot.png')
fig.show()

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

''' 10. Should you split in two groups today?'''

print('Based on the cluster analysis, our original notion of splitting up the group according to like preferences is supported. Analyzing how well separated the 10 employees would be if split into 2 separate lunch groups in the KMeans_2xClustering_PeoplePCA plot, it appears to be a good split into one group of 6 and one of 4. The first principal component has a large explained variance weight on the vegetarian preference. It appears that Mike, Skip, Kira, and Moe had high opinions on a vegetarian lunch - in fact they may even be vegetarian. However, it is also important to note that the separation is not always partitioned on this preference as Alice had a marginally stronger opinion about vegetarian options that Kira, but was believed to better fit the other groups lunch preferences when considering the other factors as well.')

#############################################################
# Cluster and Plot using PCA transformed data [People Matrix]
#############################################################
Pkmeans = KMeans(n_clusters = 2)
Pkmeans.fit(M_people_PCA)
P_centroid = Pkmeans.cluster_centers_
P_labels = Pkmeans.labels_
print('\n______________________________________')
print('#####    KMeans Clusters (2) of PCA Transformed People Matrix centroids:    #####\n')
print(P_centroid)
print('PCA Transformed People Matrix Labels: ', P_labels)

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

#https://matplotlib.org/users/colors.html
#cluster 0 is green, cluster 1 is red, cluster 2 is cyan (blue)


colors = ["g.","r."]
labelList = list(people)

for i in range(len(M_people_PCA)):
   print ("coordinate:" , M_people_PCA[i], "label:", P_labels[i])
   ax.plot(M_people_PCA[i][0],M_people_PCA[i][1],colors[P_labels[i]],markersize=10)
   #https://matplotlib.org/users/annotations_intro.html
   #https://matplotlib.org/users/text_intro.html
   ax.annotate(labelList[i], (M_people_PCA[i][0],M_people_PCA[i][1]), size=25)
ax.scatter(P_centroid[:,0],P_centroid[:,1], marker = "x", s=150, c = ['green','red'], linewidths = 5, zorder =10)
ax.set_title("KMeans People Clustering using PCA Components", size = 20)
fig.savefig('KMeans_2xClustering_PeoplePCA.png')
fig.show()


''' 11. Now you just found out the boss is paying for the meal. How should you adjust? Now what is best restaurant?'''

print('If the boss is paying for the meal, then the linear combinations should be adjusted to not include the cost as a preference for lunch since the individual employees will not need to pay for their own meal. With cost excluded, the top preference is now Amaze.')

M_RxP_BOSS = WhatsForLunch.remove_factor(2)

print('\n________________________________________________________________________________________________')
print('#####   10x10 RestaurantxPeople Matrix No Cost (Rows are Restaurants/Columns are People)     #####')
print(M_RxP_BOSS)

# Sum up scores for all people and restaurants to determine which restaurant has the highest overall score that is optimal for lunch that best accomodates everyone's preferences
RestScores_BOSS = np.sum(M_RxP_BOSS, axis = 1)
BestRestBOSS_Idx = np.argmax(RestScores_BOSS) # Eigth restraunt is best fit for the entire group (7th index)
print('\n_________________________________________________')
print('#####   Best Restaurant for Entire Office     #####')
print('\n       Optimal Restaurant (NO COST): ', list(restaurants)[BestRestBOSS_Idx])

''' 12. Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants. Can you find their weight matrix?'''

print('Combining the results and restaurant matrix, we can perform calculations to find a weight matrix for the same people because there is a single solution to this problem. However, we are unable recreate the same exact weight matrix for those same employees because theres multiple methods of solving this. In other words, using the concept of linear combinations we would be able to produce a weight matrix for peoples lunch preferences using the same restraunt matrix and each individuals optimal ranks for the restraunt, but that weight matrix may not accurately represent their preferenes because there is a large number of weight combinations that can be used to produce the same restaurant rankings.')