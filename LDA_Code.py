#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


df = pd.read_csv('winequality-red.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


target_name = 'quality'

y= df[target_name]
x= df.drop(target_name,axis=1)


# In[6]:


x.head()


# In[7]:


y.head()


# In[8]:


y.shape


# In[9]:


y.unique()


# # Observation    
#     No. of unique labels or classes = 6

# In[10]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scale = sc.fit_transform(x)


# # Split the dataset into Training set and Test set

# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.2, random_state = 0)


# In[12]:


x_train.shape, y_train.shape


# In[13]:


x_test.shape,y_test.shape


# # Feature Dimension Reduction by LDA on Wine Data
# 
# ### Assign the no. of components = No. of classes in target variable -1 
#  
# ##### No. of labels (classes) in target variable = 6
# 

# In[14]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=5)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)


# In[15]:


x_train_lda.shape, x_test_lda.shape


# In[16]:


lda.explained_variance_ratio_


# In[17]:


0.82863687 + 0.12032383 + 0.03227525 + 0.01666889 + 0.00209516


# In[18]:


colors = ['royalblue','red', 'deeppink', 'maroon', 'mediumorchid', 'tan', 'forestgreen', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x : colors[x % len(colors)])
plt.scatter(x_train_lda[:,0],x_train_lda[:,1],c=vectorizer(y_train))
plt.grid()


# ##### These 5 Linear components are separated with classes

# # Logistic Regression Model

# In[19]:


x_train_lda.shape, y_train.shape


# In[20]:


from sklearn.linear_model import  LogisticRegression
import datetime
start = datetime.datetime.now()
lg = LogisticRegression(random_state= 0)
lg.fit(x_train_lda, y_train)
end = datetime.datetime.now()
print("Total execution time with LDA  is: ", end-start)


# In[21]:


y_pred_lda = lg.predict(x_test_lda)


# In[22]:


y_pred_lda.shape


# In[23]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred_lda)
print(cm)


# In[24]:


from sklearn import metrics
print("Test accuracy on 11 features =",metrics.accuracy_score(y_test, y_pred_lda)*100)


# In[38]:


print("Classification report with LDA on 11-compoenents:\n", classification_report(y_test, y_pred_lda, digits=4))


# ### Apply Logistic Regression Model on original features (11-Features)

# In[39]:


x_train.shape, y_train.shape


# In[40]:


import datetime
start = datetime.datetime.now()
lg = LogisticRegression(random_state=0)
lg.fit(x_train,y_train)
end = datetime.datetime.now()
print("Total execution time without LDA is:", end-start)


# In[28]:


y_pred = lg.predict(x_test)


# In[29]:


cm1 = confusion_matrix(y_test, y_pred)
print(cm1)


# In[30]:


print("Test Accuracy without LDA on 11-Features = ", metrics.accuracy_score(y_test, y_pred)*100)


# In[31]:


print("Classification report without LDA on 11-Features: \n", classification_report(y_test, y_pred, digits=4))


# # Observation : Results of Logistic Regression
# 
# CASE-1 : Without LDA (11-Features):
#     1) Total execution time :
#     2) Test Accuracy :
#     
# CASE-2 : Without LDA (5-Features):
#     1) Total execution time :
#     2) Test Accuracy :
#     

# ## LDA can be used as Classification Algorithm

# In[32]:


x_train_lda.shape, y_train.shape


# In[33]:


import datetime
start = datetime.datetime.now()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda1 = LDA()
lda1.fit(x_train_lda, y_train)
end = datetime.datetime.now()
print("Total execution time of LDA on 5-components: ", end-start)


# In[34]:


y_pred_lda1 = lda1.predict(x_test_lda)


# In[35]:


print("Test accuracyh of LDA on 5 components = ", metrics.accuracy_score(y_test, y_pred_lda1)*100)


# In[36]:


cm3 = confusion_matrix(y_test, y_pred_lda1)
cm3


# In[37]:


print("Classification report of LDA on 5-components : \n", classification_report(y_test, y_pred_lda1,digits=4))


# In[ ]:




