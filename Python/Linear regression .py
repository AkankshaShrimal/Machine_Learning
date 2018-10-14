
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import numpy as np


# In[5]:


#importing datasets from scikit learn
from sklearn import datasets , linear_model


# In[6]:


#load the dataset
house_price = [245, 312, 279, 308, 199, 219, 405, 325, 319, 255]
size = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]


# In[7]:


#reshape the input to your regression
# -1 means create a arrray and 1 means each having 1 element. all elements in size2 are [[1400][1600] ..] 
size2 = np.array(size).reshape((-1,1))


# In[8]:


#linear regression classifier
regr = linear_model.LinearRegression()


# In[9]:


#fit module used to fit data frequently and quickly
regr.fit(size2, house_price)


# In[10]:


#printing coefficient and intercept

print("coefficient : \n", regr.coef_ )
print("intercept : \n", regr.intercept_)


# In[11]:


# checking prediction by formula a + b(size) = price
size_new = 1400
price = regr.coef_ * size_new + regr.intercept_
print("by formula ", price)
# By inbuilt function predict
print(regr.predict([[size_new]]))


# In[12]:


#Formula obtained for the trained model
def graph(formula, x_range):
    # x_range array converted to np array
    x = np.array(x_range)
    # formula is evaluated as y
    y = eval(formula)
    #plotting graph
    plt.plot(x, y)


# In[13]:


#Plotting the prediction line
graph('regr.coef_*x + regr.intercept_', range(1000,2700))
plt.scatter(size, house_price, color='black')
plt.ylabel('House Price')
plt.xlabel('Size of Houses')
plt.show()

