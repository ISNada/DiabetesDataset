#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


diabetes = pd.read_csv('\diabetes.csv')
print(diabetes.columns)


# In[5]:


diabetes.head()


# In[6]:


#The diabetes data set consists of 768 data points, with 9 features each
print("dimension of diabetes data: {}".format(diabetes.shape))


# In[7]:


#“Outcome” is the feature we are going to predict,
#0 means No diabetes, 1 means diabetes. Of these 768 data points,
#500 are labeled as 0 and 268 as 1:
print(diabetes.groupby('Outcome').size())


# In[18]:


import seaborn as sns
sns.countplot(diabetes['Outcome'],label="Count")


# In[17]:


diabetes.info()


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'],
                                                    diabetes['Outcome'], stratify=diabetes['Outcome'],
                                                    random_state=66)


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')


# In[11]:


knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))


# In[ ]:




