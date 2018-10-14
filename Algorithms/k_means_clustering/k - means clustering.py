
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


from matplotlib import style
style.use('ggplot')


# In[5]:


from sklearn.cluster import KMeans


# In[7]:


# Plotting and visualizing data before feeding into ML algorithm
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]
plt.scatter(x,y)
plt.show()


# In[10]:


# Converting data to a numpy array
X = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])


# In[11]:


# Initialise k means by number of clusters
kmeans = KMeans(n_clusters = 2) 


# In[12]:


# Fitting data in kMeans
kmeans.fit(X)


# In[13]:


# Getting the values of centroids
centroids = kmeans.cluster_centers_
print(centroids)


# In[14]:


#Gettig labes that is to which cluster centroid does each point map to in sequence of data given
labels = kmeans.labels_
print(labels)


# In[34]:


# Plotting and visualizing output
# Defining colors
colors = ["g.", "r.", "c.", "y."]

# Plotting points with different colors according to cluster- green  for 0 , red for 1
for i in range(len(X)):
    print("coordinate: " ,X[i], " label: ", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
# Plotting cetroids for each cluster
plt.scatter(centroids[:, 0], centroids[:, 1], marker = "X" , s=120 , linewidths=1, zorder = 10)

plt.show()
    

