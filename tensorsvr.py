
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

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


pml_train_y.describe()


# In[6]:
# In[7]:


pml_test = pd.read_csv('pml_test_features.csv')
pml_test_X = pml_test.drop(columns=['id'])
pml_test_X.shape


# In[12]:


def col_op(col):
    return col.astype('category').cat.codes


# In[13]:


def digit_op(code):
    num = 0
    for alpha in code:
        num *= 26
        num += ord(alpha) - ord('A') + 1
    return num

def to_digit(col):
    return col.apply(digit_op)


# In[14]:


to_digit(pd.Series(['AA','BB','A']))


# In[15]:


def encode_X_to_digit(orig_train_X):
    c_X = orig_train_X.copy()
    tmp = orig_train_X.select_dtypes(exclude=['float64','int64'])
    # tmp = pd.Categorical(tmp)

    c_X.loc[:, tmp.columns] = tmp.apply(lambda col: col.astype('category').cat.codes)
    return c_X


# In[16]:


pml_train_c_X = encode_X_to_digit(pml_train_X)
pml_test_c_X = encode_X_to_digit(pml_test_X)
print('data encoded')


# In[17]:


pml_train_c_X.head()


# In[18]:


# #Generate a correlation matrix between features
# x_corr = pml_train_c_X.corr()
# mask = np.zeros_like(x_corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# f, ax = plt.subplots(figsize=(22, 18))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.heatmap(x_corr, mask=mask, vmax=1, cmap=cmap, center=0,
#             square=True, linewidths=.5)


# In[19]:


# # standardization
# scaler = preprocessing.StandardScaler()
# pml_train_c_X = scaler.transform(pml_train_c_X)
# pml_test_c_X = scaler.transform(pml_test_c_X)

# pml_test_c_X.shape


# In[20]:


# normalize
normalizer = MinMaxScaler().fit(pml_train_c_X)
pml_train_c_X = normalizer.transform(pml_train_c_X)
pml_test_c_X = normalizer.transform(pml_test_c_X)

print(pml_test_c_X.shape)


# In[21]:


# kBest = SelectKBest( k=50)
# pml_train_sfs_X = kBest.fit_transform(pml_train_c_X, pml_train_y)
# pml_test_sfs_X = kBest.transform(pml_test_c_X)


# In[22]:


# pml_test_sfs_X.shape


# In[23]:


# test/train split
split_train_X = pml_train_c_X
split_test_X = pml_test_c_X
val_train_X, val_test_X, val_train_y, val_test_y = train_test_split(split_train_X, pml_train_y, random_state=2018, test_size=0.05)
print('data prep finished')


# In[24]:


def mse_score(clf, X, y):
    prd_y = clf.predict(X)
    return np.sqrt(np.sum((prd_y-y)**2)/len(y))


# In[25]:


# pipeline:
def try_clf(clf, clf_name='', save=False, train_X = val_train_X, train_y = val_train_y, test_X = val_test_X, test_y = val_test_y):
    print('start training ' + clf_name )
    clf.fit(train_X, train_y)
    if save == True:        
        from sklearn.externals import joblib
        joblib.dump(clf, '%s.pkl'%clf_name) 
    print( str(train_X.shape) + "data trained," + str(test_X.shape) + " data tested")
    print(clf_name + ' train :' + str(clf.score(train_X, train_y)))
    print(clf_name + ' test  :' + str(clf.score(test_X, test_y)))
    print(mse_score(clf, test_X, test_y))
    
    return clf


# In[26]:


def use_clf(clf, clf_name='clf', pml_X = split_test_X):
    ans = clf.predict(pml_X)
    filename = clf_name + '.csv'
    pd.DataFrame({'id':pml_test['id'], 
          'loss':ans}).to_csv(filename,index = False)
    print('exported as ' + filename)


# ### Tuning

# In[27]:


# def tune_para_my(clf, train_X, train_y, test_X, test_y):
#     clf = try_clf(clf, 'svr rbf',svr_train_X,  svr_train_y, svr_test_X, svr_test_y)


# In[28]:

# ## Linear

# ## SVR

# In[31]:


var_keep = np.array([False, False, False, False, False, False, False,  True, False,
       False, False, False, False, False,  True, False, False,  True,
       False, False,  True,  True, False, False, False, False, False,
       False,  True,  True,  True, False, False, False, False, False,
       False, False,  True, False, False,  True, False, False,  True,
       False, False, False, False, False, False, False, False,  True,
       False,  True, False, False, False,  True, False, False, False,
        True, False, False, False,  True,  True,  True, False, False,
       False, False, False, False,  True, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False])


# In[32]:


svr_X = pml_train_c_X[:,var_keep==False]
svr_y = pml_train_y
pml_svr_test_X = pml_test_c_X[:,var_keep==False]

# svr_train_X, svr_test_X, svr_train_y, svr_test_y = train_test_split(svr_X, svr_y, random_state=2018, test_size=0.05)


# In[33]:


idx = np.random.choice(svr_X.shape[0], 10000, replace=False)
svr_X = svr_X[idx, :]
svr_y = svr_y.reshape(-1,1)[idx, :].squeeze()


# In[46]:


svr_train_X, svr_test_X, svr_train_y, svr_test_y = train_test_split(svr_X, svr_y, random_state=2018, test_size=0.2)


# In[35]:


svr_X.shape


# In[36]:


svr_sigmoid = SVR(kernel='sigmoid', C=0.1, cache_size=300)
svr_sigmoid = try_clf(svr, 'svr rbf',False, svr_train_X,  svr_train_y, svr_test_X, svr_test_y)


# In[37]:


np.array([3,2,44,3]).argmax()


# In[38]:


use_clf(svr,'svr_sigmoid',pml_svr_test_X)


# In[40]:


pml_train_y.shape


# In[43]:

# In[41]:


# In[47]:



print('------final run-------')
clf_name = 'tuned_svr'
tuned_svr = SVR( degree = 2, gamma = 0.08, C=0.1, tol = 0.1, cache_size=300)
tuned_svr = try_clf(tuned_svr, clf_name, True)
use_clf(tuned_svr,clf_name)


# In[ ]:


# import pickle
# s = pickle.dumps(clf)
# clf2 = pickle.loads(s)


# In[ ]:


from sklearn.externals import joblib
joblib.dump(tuned_svr, 'tuned_svr.pkl') 
