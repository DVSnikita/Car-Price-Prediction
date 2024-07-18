#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df = pd.read_csv("Car Dataset.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


# Removing Duplicated 
df = df.drop_duplicates()


# In[9]:


df.duplicated().sum()


# In[10]:


# Data after droppind duplicated values 
df.shape


# In[11]:


# Reset indexing
df.reset_index(drop = True, inplace = True )


# In[12]:


df.head(3)


# In[13]:


df.tail(2)


# In[14]:


df.dtypes


# In[ ]:





# In[15]:


# Convert Calegorical Variables into numerical variables using LableEncoder

le = LabelEncoder()
df['name'] = le.fit_transform(df["name"])
df['fuel'] = le.fit_transform(df["fuel"])
df['seller_type'] = le.fit_transform(df["seller_type"])
df['transmission'] = le.fit_transform(df["transmission"])
df['owner'] = le.fit_transform(df["owner"])


# In[16]:


df.head()


# In[ ]:





# In[17]:


ss = StandardScaler()
df = pd.DataFrame(ss.fit_transform(df),columns = df.columns)


# In[18]:


df.head()


# In[ ]:





# In[ ]:





# In[19]:


# Split the dataset
X = df.drop('selling_price', axis=1)
y = df['selling_price']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


# Bayesian Regression
bayesian_model = BayesianRidge()
bayesian_model.fit(X_train, y_train)
bayesian_preds = bayesian_model.predict(X_test)


# In[22]:


print("Bayesian Regression MSE:", mean_squared_error(y_test, bayesian_preds))
print("Bayesian Regression R2:", r2_score(y_test, bayesian_preds))


# In[23]:


# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


# In[24]:


print("Random Forest MSE:", mean_squared_error(y_test, rf_preds))
print("Random Forest R2:", r2_score(y_test, rf_preds))


# In[25]:


# Gradient Boosting
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)


# In[26]:


print("Gradient Boosting MSE:", mean_squared_error(y_test, gb_preds))
print("Gradient Boosting R2:", r2_score(y_test, gb_preds))


# In[ ]:




