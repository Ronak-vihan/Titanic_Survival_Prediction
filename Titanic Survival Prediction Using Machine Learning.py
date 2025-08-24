#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image


# In[98]:


titanic_df = pd.read_csv("Titanic-Dataset.csv")
titanic_df.head()


# In[99]:


titanic_df.shape


# # Exploration Part

# In[100]:


titanic_df.info()


# In[101]:


import seaborn as sns
sns.heatmap(titanic_df.isnull(), cmap='plasma')


# In[102]:


titanic_df.isnull().sum()


# In[103]:


titanic_df.drop('Cabin',axis=1, inplace=True)


# In[104]:


titanic_df.head()


# In[105]:


titanic_df['Embarked'].value_counts()


# In[106]:


titanic_df.dropna(inplace=True)
titanic_df.isnull().sum()


# # Transformation into a categorical column.

# In[107]:


print(titanic_df['Sex'].unique())
print(titanic_df['Embarked'].unique())


# In[108]:


titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
titanic_df['Embarked'] = titanic_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


# In[109]:


titanic_df.head(5)


# # Let's split the data into the target and feature variables.

# In[110]:


x = titanic_df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis = 1) 
y = titanic_df['Survived']


# In[111]:


x.shape


# In[112]:


y.shape


# # Train test split

# In[113]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=2)


# In[114]:


x_train.shape


# # Traininig Model

# In[120]:


lg = LogisticRegression()
lg.fit(x_train, y_train)
y_pred = lg.predict(x_test)
print(accuracy_score(y_test, y_pred))


# In[121]:


y_pred


# # Checking for a Random Person

# In[126]:


input_df = (2, 	0, 	27.0, 	0, 	0, 	13.0000, 	0 	)
input_df_np = np.asarray(input_df)
prediction = lg.predict(input_df_np.reshape(1, -1))
if prediction[0] == 0:
    print('This person Survived')
else:
    print('This person Dead')


# In[125]:


x


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




