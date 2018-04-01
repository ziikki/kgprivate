
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# In[3]:


pml = pd.read_csv('pml_train.csv')
print('data loaded')
pml.head()


# In[4]:


pml_train_y = pml.loss
pml_train_X = pml.drop(columns=['id', 'loss'])
pml_train_X.shape


# In[5]:


pml_train_d_X = pd.get_dummies(pml_train_X)
pml_train_d_X.shape


# In[6]:


pml_test = pd.read_csv('pml_test_features.csv')
pml_test_X = pml_test.drop(columns=['id'])
pml_test_X.shape


# In[7]:


# # one-hot encoded test
# pml_test_d_X = pd.get_dummies(pml_test)
# pml_test_d_X.shape


# In[8]:


# cDrop = [c for c in pml_test_d_X.columns if c not in pml_train_d_X.columns]
# print(cDrop)
# pml_test_d_X.drop(columns = cDrop, inplace=True)

# for c in pml_train_d_X.columns:
#     if c not in pml_test_d_X.columns:
#         pml_test_d_X[c] = 0
# print(pml_test_d_X.shape)
# pml_test_d_X.head()


# In[9]:


# pml_train_cont = pml_train_X.filter(regex=("cont\d*"))
# pml_train_cont.head()


# In[10]:


#Generate a correlation matrix between features
# x_corr = pml_train_cont.corr()
# mask = np.zeros_like(x_corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# f, ax = plt.subplots(figsize=(11, 9))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.heatmap(x_corr, mask=mask, vmax=1, cmap=cmap, center=0,
#             square=True, linewidths=.5)


# In[11]:


def col_op(col):
    return col.astype('category').cat.codes


# In[12]:


def digit_op(code):
    num = 0
    for alpha in code:
        num *= 26
        num += ord(alpha) - ord('A') + 1
    return num

def to_digit(col):
    return col.apply(digit_op)


# In[13]:


to_digit(pd.Series(['AA','BB','A']))


# In[14]:


def encode_X_to_digit(orig_train_X):
    c_X = orig_train_X.copy()
    tmp = orig_train_X.select_dtypes(exclude=['float64','int64'])
    # tmp = pd.Categorical(tmp)

    c_X.loc[:, tmp.columns] = tmp.apply(lambda col: col.astype('category').cat.codes)
    return c_X


# In[27]:


pml_train_c_X = encode_X_to_digit(pml_train_X)
pml_test_c_X = encode_X_to_digit(pml_test_X)
print('data encoded')


# In[28]:


pml_train_c_X.head()


# In[29]:


# #Generate a correlation matrix between features
# x_corr = pml_train_c_X.corr()
# mask = np.zeros_like(x_corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# f, ax = plt.subplots(figsize=(22, 18))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.heatmap(x_corr, mask=mask, vmax=1, cmap=cmap, center=0,
#             square=True, linewidths=.5)


# In[30]:


# normalize
normalizer = Normalizer().fit(pml_train_c_X)
pml_train_c_X = normalizer.transform(pml_train_c_X)
pml_test_c_X = normalizer.transform(pml_test_c_X)
pml_test_c_X.shape


# In[31]:


# test/train split
val_train_X, val_test_X, val_train_y, val_test_y = train_test_split(pml_train_c_X, pml_train_y, random_state=2018, test_size=0.05)
print('data prep finished')


# In[20]:


# pipeline:
def try_clf(clf, clf_name=''):
    print('start training ' + clf_name )
    clf.fit(val_train_X, val_train_y)
    print(clf_name + ' train :' + str(clf.score(val_train_X, val_train_y)))
    print(clf_name + ' test  :' + str(clf.score(val_test_X, val_test_y)))
    
    return clf


# In[34]:


def use_clf(clf, clf_name='clf'):
    ans = clf.predict(pml_test_c_X)
    filename = clf_name + '.csv'
    pd.DataFrame({'id':pml_test['id'], 
          'loss':ans}).to_csv(filename,index = False)
    print('exported as ' + filename)


# ## Linear

# In[32]:


lin = LinearRegression()
lin = try_clf(lin)


# In[35]:


use_clf(lin, 'lin_default')


# ## SVR

# In[ ]:


svr = SVR(C=1.0, epsilon=0.2)
svr = try_clf(svr, 'svr rbf')


# In[37]:


use_clf(svr,'svr_default')


# ## tree

# In[ ]:


rf = RandomForestRegressor(n_estimators=10,criterion = 'mae',
                             max_leaf_nodes = 5, random_state = 2018)
rf = try_clf(rf, 'rf')


# In[ ]:


use_clf(rf,'rf_10tree_5maxnodes')


# In[ ]:


c = 'AB'


# In[ ]:


for a in reversed(c):
    print(a)


# In[30]:


temp = pd.DataFrame({'a':['A','B','C','B'],
             'b':['DC','BD','BF','CC']})
temp


# In[31]:


def op(col):
    return col.astype('category').cat.codes


# In[39]:


# uniques = np.sort(pd.unique(temp.values.ravel()))
# temp.apply(lambda x: x.astype('category', categories=uniques))

le = LabelEncoder()
le.fit(temp.values.flat)

# Convert to digits.
temp = temp.apply(le.transform)
temp

