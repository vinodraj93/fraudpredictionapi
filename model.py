#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np 
import pandas as pd 


# In[3]:


data = pd.read_csv('datasets_156197_358170_Churn_Modelling.csv')
data.head(5)


# In[4]:


data.shape


# In[5]:


data.isna().sum()


# In[6]:


len(set(data['CustomerId']))


# # Average Tenure

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
import statistics


# In[8]:


def distribution(da):
    if da not in ['RowNumber','CustomerId']:
        a_string = list(str(type(data[da][0])).split('.'))
        matches = ["int64'>","float64'>"]
        if len([x for x in matches if x in a_string]) >=1:
            average = statistics.mean(data[da])
            med = statistics.median(data[da])
            mode = statistics.mode(data[da])
            sns.distplot(data[da])
            plt.axvline(x=average, c='orange',label='Average')
            plt.axvline(x=med,c='red',label='50% of sample')
            plt.axvline(x=mode,c='blue',label='most common')
            plt.title(da)
            plt.legend()
            plt.show()
            
            sns.boxplot(data[da])
            plt.title(da)
            plt.show()
            
            print(data[da].describe())


# In[9]:


# plt.figure(figsize=(15,10))
for i in data.columns:
    distribution(i)
    
    



# In[10]:


listoftenure = list(round(data.groupby('Tenure').count().iloc[:,0]*100/len(data),2))

plt.bar(range(len(listoftenure)),listoftenure)


# In[11]:


X = data.iloc[:, 3:-1]
y = data.iloc[:, -1]
X.head(5)


# In[12]:


def data_encode(df):
    df = pd.get_dummies(data = df, columns=["Geography"], drop_first = False)
    for col in df.select_dtypes(include=['category','object']).columns:
        codes,_ = df[col].factorize(sort=True)    
        df[col]=codes
    return df


# In[13]:


X = data_encode(X)
X.head(5)


# In[14]:


from sklearn.model_selection import train_test_split
 # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[15]:


from sklearn.ensemble import RandomForestClassifier
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[16]:


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[17]:


import pickle
pickle.dump(clf, open('model.pkl', 'wb')) 

