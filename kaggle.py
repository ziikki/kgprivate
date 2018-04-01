
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[1]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[3]:


pml = pd.read_csv('pml_train.csv')


# In[4]:


pml.head()


# In[5]:


pml.describe()


# In[6]:


pml.columns


# In[7]:


pml_train_y = pml.loss
pml_train_X = pml.drop(columns=['id', 'loss'])
pml_train_X.shape


# In[8]:


pml_train_d_X = pd.get_dummies(pml_train_X)
pml_train_d_X.shape


# In[9]:


pml_test = pd.read_csv('pml_test_features.csv')
pml_test_X = pml_test.drop(columns=['id'])
pml_test_X.shape


# In[10]:


# one-hot encoded test
pml_test_d_X = pd.get_dummies(pml_test)
pml_test_d_X.shape


# In[11]:


cDrop = [c for c in pml_test_d_X.columns if c not in pml_train_d_X.columns]
print(cDrop)
pml_test_d_X.drop(columns = cDrop, inplace=True)

for c in pml_train_d_X.columns:
    if c not in pml_test_d_X.columns:
        pml_test_d_X[c] = 0
print(pml_test_d_X.shape)


# In[12]:


pml_test_d_X.head()


# In[13]:


pml_test.head()


# In[16]:


a = 3


# Feature Engineering

# In[14]:


pml_train_cont = pml_train_X.filter(regex=("cont\d*"))
pml_train_cont.head()


# ## Train

# In[ ]:


train_d_X, test_d_X, train_d_y, test_d_y =   train_test_split(pml_train_d_X, pml_train_y, random_state=2018, test_size=0.2)


# In[ ]:


regr1 = RandomForestRegressor(n_estimators=20,criterion = 'mae',
                             max_leaf_nodes = 5, random_state = 2018)
regr1.fit(train_d_X, train_d_y)


# In[ ]:
print("regr1")
print(regr1.score(train_d_X, train_d_y))
print(regr1.score(test_d_X, test_d_y))


# In[ ]:


result = regr1.predict(pml_test_d_X)
result.shape


# In[ ]:


pml_test.shape


# In[ ]:


pml_ans1 = pd.DataFrame({'id' : pml_test['id'],
                       'loss': result})
pml_ans1.head()


# In[ ]:


pml_ans1.to_csv('./answer1.csv',index=False)


# In[ ]:


regr2 = RandomForestRegressor(n_estimators=40,criterion = 'mae',
                             max_leaf_nodes = 2, random_state = 2018)
regr2.fit(train_d_X, train_d_y)


# In[ ]:

print("regr2")
print(regr2.score(train_d_X, train_d_y))
print(regr2.score(test_d_X, test_d_y))


# In[ ]:


result2 = regr2.predict(pml_test_d_X)
result2.shape


# In[ ]:


pml_ans2 = pd.DataFrame({'id' : pml_test['id'],
                       'loss': result2})
pml_ans2.head()


# In[ ]:


pml_ans2.to_csv('./answer2.csv',index=False)

