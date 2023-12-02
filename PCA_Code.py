#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score


# In[40]:


df = pd.read_csv('data.csv')


# In[41]:


df.shape


# In[42]:


df.head()


# In[43]:


df.info()


# In[44]:


df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
df.diagnosis.value_counts()


# In[45]:


X = df.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]]
y = df.diagnosis.values


# In[46]:


X,y


# In[62]:


#Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state = 4)


# In[63]:


from sklearn.linear_model import LogisticRegression

classifier =  LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn import metrics

cm = confusion_matrix(y_test,y_pred)
print(cm)

print("Accuracy :",metrics.accuracy_score(y_test, y_pred)*100)


# In[85]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state = 0)


#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(X)


# In[86]:


#Applying PCA


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[87]:


explained_variance = pca.explained_variance_ratio_
explained_variance


# In[1]:


0.98187084 + 0.01627803


# In[88]:


#Training the logistic regression model

from sklearn.linear_model import LogisticRegression
classifier =  LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)


# In[89]:


#Prediction part

y_pred = classifier.predict(X_test)


# In[91]:


import seaborn as sns
sns.heatmap(cm,annot=True)


# In[90]:


#Making a confusion matrix

from sklearn.metrics import confusion_matrix
from sklearn import metrics

cm = confusion_matrix(y_test,y_pred)
print(cm)

print("Accuracy :",metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:





# In[ ]:




