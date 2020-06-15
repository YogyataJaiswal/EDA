#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')


# In[4]:


df.head()


# In[6]:


df.shape


# ## Univarient Analysis

# In[14]:


df_setosa = df.loc[df['species'] =='setosa']


# In[15]:


df_virginica=df.loc[df['species']=='virginica']
df_versicolor=df.loc[df['species']=='versicolor']


# In[16]:


plt.plot(df_setosa['sepal_length'],np.zeros_like(df_setosa['sepal_length']),'o')
plt.plot(df_virginica['sepal_length'],np.zeros_like(df_virginica['sepal_length']),'o')
plt.plot(df_versicolor['sepal_length'],np.zeros_like(df_versicolor['sepal_length']),'o')
plt.xlabel('Petal length')
plt.show()


# ## Bivariet Analysis

# In[18]:


sns.FacetGrid(df,hue="species",size=5).map(plt.scatter,"petal_length","sepal_width").add_legend();
plt.show();


# ## Multivariate Analysis

# In[21]:


sns.pairplot(df,hue="species",size=3)


# In[ ]:




