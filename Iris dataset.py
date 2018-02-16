
# coding: utf-8

# In[1]:


import sys
sys.version


# In[2]:


# import load_iris from dataset module in sklearn
from sklearn.datasets import load_iris


# In[3]:


#save 'bunch' object containing the iris dataset and attributes in a variable
iris = load_iris()
type(iris)


# In[4]:


#print iris data set
print (iris.data)


# # ML Terminology
#     
#     -each Row is an observation or sample,example,instance,record
#     -each Column is a feature or predictor,attribute,independent variable,input,regressor,covariate

# In[5]:


#print the names of four features or columns
print (iris.feature_names)


# In[6]:


#print integers representing the species of each obs
print (iris.target)
#represents what we are going to predict


# In[7]:


#print the encoding scheme for species 0 = setosa 1 = versicolor 2 = virginica
print(iris.target_names)


# -Each value that is being predicted is the response aka target,outcome,label,dependent variable
# -classification is supervised learning which the response is categorical its values are in a finite un-ordered set
# -regression is a supervised learning which response is ordered and continuous example(price of house)
# 
# must first decide how data is encoded then figure out if classification or regression tech is right
# 

# # First step in ML
# -get comp to learn relationship between features and response
# must be in form sklearn expects in 4 different ways
# 1) passed in ML model as seperate objects
# 2) sklearn only expects to see numbers in features and response target, should always be numberic in reg or class
# 3) expects to be stored as numpy arrays
# 4) Feaeture and response obj should be certain shapes 2D
# 
# numpy has shape attribute to verify the shape of data
# response obj expected to have single dimension and should have same mag as feature obj. One response to each attribute

# In[8]:


print(iris.data)


# In[9]:


print (iris.target.shape)


# In[10]:


print (iris.data.shape)


# In[11]:


X = iris.data
#uppercase X represent a matrix
y = iris.target
#lowercase y represent a label


# # Training a ML model

# - Bc the response is categorical this is a categorical question(for IRIS dataset)
# - KNN is the K-nearest neighbor a comparitive training model or context clues. Takes n-values nearest to the UNKNOWN data and uses the most popular response as the predictive basis
# 1) First pick a value for K

# In[12]:


print(X.shape)
print(y.shape)


# When inputing data make sure to meet all 4 key reqs

# # Sklearn 4 step modeling pattern
# 

# * Step 1 import class you plan to use from sklearn
# from sklearn.neighbors import KNeighborsClassifier

# In[13]:


from sklearn.neighbors import KNeighborsClassifier


# * Step 2 "Instantiate" the "estimator"
# -estimator is sklearn model
# - this process is instantiation or "make an instance of"

# In[14]:


knn = KNeighborsClassifier(n_neighbors=1)
#this is an instance, now have an object knn that knows how to do KNN


# 3 notes
# doesnt matter name of instantiator
# specify the argument(n_neighbors) to tell knn object
# all parameters not specified are in default

# In[15]:


print (knn)


# * Step 3 "Fit Model with data"
# model learns relation btwn x and y
# happens in place

# In[16]:


knn.fit(X, y)


# * Step 4 predict the response
# -New obs are called "out of sample" data
# - uses info it learned during model training process to predict an outcome

# In[17]:


import numpy as np
X_new = [[3, 5, 4, 2]]
#needed to reshape from 1D array to a 2D array. (1,-1) tells 
#Numpy to infer the dimensions to be 1 for the first dimension and then 4 for the second
knn.predict(X_new)


# - predict method does have a response object in the form of array
# - ML doesn't know what array[2] is and therefore is up to us to recall that array 2 is Virginica

# In[18]:


X_newer = [[3,4,5,2], [5,4,3,2]]
knn.predict(X_newer)


# # Using a different Value for K

# In[19]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
knn.predict(X_newer)


# Sklearn models have uniform interface and can still use the same 4 steps

# In[20]:


#import class LogisticsRegression
from sklearn.linear_model import LogisticRegression

#instantiate the model
loreg = LogisticRegression()

#fit the data to the model
loreg.fit(X,y)

#make a prediction
loreg.predict(X_newer)


# These are out of sample measurment and therefore do not know true response value and would not know which MODEL,KNN or LogisticRegression worked best.
# - With Supervised learning want to build models that generalize to new data
# -Compare models to existing label data to choose which model is best

# # Determining Best Model

# Model Eval Procedure
# 1)Train and Test
# The most basic and no brainer training is to have a given dataset with KNOWN values, split it up RANDOMLY into two datasets(train and test) and use each split dataset for their respective names.
# This is to troubleshoot if the model is accurate when given outside attributes

# In[30]:


from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
irisdf_x = pd.DataFrame(iris.data, columns=iris.feature_names)
irisdf_y = pd.DataFrame(iris.target)


# In[28]:


irisdf_x.describe()


# In[29]:


irisdf_y.describe()


# In[31]:


reg = linear_model.LinearRegression()
xtrain, xtest, ytrain, ytest = train_test_split(irisdf_x, irisdf_y, test_size = .33, random_state = 25)


# In[32]:


reg.fit(xtrain,ytrain)


# In[33]:


reg.coef_


# In[37]:


a = reg.predict(xtest)
a[2]


# In[38]:


ytest


# In[39]:


reg.score(xtest,ytest)


# Probably best to use another testing model
