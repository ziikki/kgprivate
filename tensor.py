
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


pml_train_d_X = pd.get_dummies(pml_train_X)
pml_train_d_X.shape


# In[7]:


pml_test = pd.read_csv('pml_test_features.csv')
pml_test_X = pml_test.drop(columns=['id'])
pml_test_X.shape


# In[8]:


# # one-hot encoded test
# pml_test_d_X = pd.get_dummies(pml_test)
# pml_test_d_X.shape


# In[9]:


# cDrop = [c for c in pml_test_d_X.columns if c not in pml_train_d_X.columns]
# print(cDrop)
# pml_test_d_X.drop(columns = cDrop, inplace=True)

# for c in pml_train_d_X.columns:
#     if c not in pml_test_d_X.columns:
#         pml_test_d_X[c] = 0
# print(pml_test_d_X.shape)
# pml_test_d_X.head()


# In[10]:


# pml_train_cont = pml_train_X.filter(regex=("cont\d*"))
# pml_train_cont.head()


# In[11]:


#Generate a correlation matrix between features
# x_corr = pml_train_cont.corr()
# mask = np.zeros_like(x_corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# f, ax = plt.subplots(figsize=(11, 9))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.heatmap(x_corr, mask=mask, vmax=1, cmap=cmap, center=0,
#             square=True, linewidths=.5)


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
def try_clf(clf, clf_name='', train_X = val_train_X, train_y = val_train_y, test_X = val_test_X, test_y = val_test_y):
    print('start training ' + clf_name )
    clf.fit(train_X, train_y)
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


def tune_para_cv(estimator, X, y, parameter_name ,parameters, k_fold = 10):
    
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV
    
    tuned_parameters = [{parameter_name: parameters}]
    n_folds = k_fold

    clf = GridSearchCV(estimator, tuned_parameters, cv=n_folds, refit=False)
    clf.fit(X, y)
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']
    plt.figure().set_size_inches(8, 6)

    plt.plot(parameters, scores)

    # plot error lines showing +/- std. errors of the scores
    std_error = scores_std / np.sqrt(n_folds)

    plt.plot(parameters, scores + std_error, 'b--')
    plt.plot(parameters, scores - std_error, 'b--')

    # alpha=0.2 controls the translucency of the fill color
    #plt.fill_between(cc, scores + std_error, scores - std_error, alpha=0.2)

    plt.ylabel('CV score +/- std error')
    plt.xlabel(parameter_name)
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.xlim([parameters[0], parameters[-1]])

    plt.show()
    
def print_cv(scores):
    print(scores)
    print('Mean:\t %f' % np.mean(scores))
    print('Var :\t %f' % np.var(scores))


# ## Linear

# In[29]:


lin = LinearRegression()
lin = try_clf(lin)


# In[30]:


use_clf(lin, 'lin_default')


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


svr = SVR(kernel='poly', gamma = 0.01, C=0.1, degree=2, tol = 0.1, cache_size=300)
svr = try_clf(svr, 'svr rbf',svr_train_X,  svr_train_y, svr_test_X, svr_test_y)


# In[37]:


np.array([3,2,44,3]).argmax()


# In[38]:


use_clf(svr,'svr_default',pml_svr_test_X)


# In[40]:


pml_train_y.shape


# In[43]:


svr_degree_scores = []
degree_range = np.arange(3,6)
for dgr in degree_range:
    print('------degree '+ str(dgr) +'-------')
    clf_name = 'svr_'+str(dgr)+'_degree'
    degree_svr = SVR( degree = dgr, C=0.1, tol = 0.1, cache_size=300)
    degree_svr = try_clf(degree_svr, clf_name,svr_train_X,  svr_train_y, svr_test_X, svr_test_y)
    svr_degree_scores.append(mse_score(degree_svr, svr_test_X, svr_test_y))
    use_clf(svr,clf_name,pml_svr_test_X)
    
best_degree = degree_range[svr_degree_scores.argmin()]


# In[44]:


svr_degree_scores


# In[41]:


np.logspace(-8,-2,num = 5,base=2)


# In[45]:


svr_degree_scores = []
degree_range = np.arange(2,5)
for dgr in degree_range:
    print('------degree '+ str(dgr) +'-------')
    clf_name = 'svr_'+str(dgr)+'_degree'
    degree_svr = SVR( degree = dgr, C=0.1, tol = 0.1, cache_size=300)
    degree_svr = try_clf(degree_svr, 'svr rbf',svr_train_X,  svr_train_y, svr_test_X, svr_test_y)
    print(degree_svr)
    svr_degree_scores.append(mse_score(degree_svr, svr_test_X, svr_test_y))
    use_clf(svr,clf_name,pml_svr_test_X)
    


# In[47]:


svr_degree_scores


# In[48]:


svr_gamma_scores = []
gamma_range = np.logspace(-8,-2,num = 5,base=2)
for gm in gamma_range:
    print('------gamma '+ str(gm) +'-------')
    clf_name = 'svr_'+str(gm)+'_gamma'
    gm_svr = SVR( degree = 2, gamma = 0.08, C=0.1, tol = 0.1, cache_size=300)
    gm_svr = try_clf(gm_svr, clf_name,svr_train_X,  svr_train_y, svr_test_X, svr_test_y)
    print(gm_svr)
    svr_gamma_scores.append(mse_score(gm_svr, svr_test_X, svr_test_y))
    use_clf(gm_svr,clf_name,pml_svr_test_X)


# In[49]:


svr_gamma_scores


# In[50]:


best_gamma = gamma_range[svr_gamma_scores.index(min(svr_gamma_scores))]


# In[ ]:


gamma_range = np.logspace(-8,-2,num = 5,base=2)

clf = GridSearchCV(estimator, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']


# In[ ]:


print('------final run-------')
clf_name = 'tuned_svr'
tuned_svr = SVR( degree = 2, gamma = 0.08, C=0.1, tol = 0.1, cache_size=300)
tuned_svr = try_clf(tuned_svr, clf_name)
use_clf(tuned_svr,clf_name)


# In[ ]:


# import pickle
# s = pickle.dumps(clf)
# clf2 = pickle.loads(s)


# In[ ]:


from sklearn.externals import joblib
joblib.dump(tuned_svr, 'tuned_svr.pkl') 


# ## tree

# In[ ]:


#. ec2-spotter/fast_ai/create_vpc.sh


# In[ ]:


rf = RandomForestRegressor(n_estimators=20, max_leaf_nodes = 20, 
                            random_state = 2018)
rf = try_clf(rf, 'rf',,,,,)


# In[ ]:


use_clf(rf,'csv/rf_20tree_20maxnode')


# In[ ]:


rf2 = RandomForestRegressor(n_estimators=10, max_leaf_nodes = 100, 
                            random_state = 2018)
rf2 = try_clf(rf2, 'rf2')


# In[ ]:


use_clf(rf2,'csv/rf_10tree_100maxnode')


# In[ ]:


rf3 = RandomForestRegressor(n_estimators=30, max_leaf_nodes = 120, 
                            min_samples_split=10, random_state = 2018)
rf3 = try_clf(rf3, 'rf3')


# In[ ]:


use_clf(rf3,'csv/rf_30tree_120maxnode')


# In[ ]:


temp_prd_y = rf3.predict(val_train_y)
temp_y = val_test_y
plt.scatter(temp_prd_y,temp_y)


# In[ ]:


rf3.feature_importances_ == 0


# In[ ]:


rf4 = RandomForestRegressor(n_estimators=50, max_leaf_nodes = 100, 
                             min_impurity_decrease=0.1, random_state = 2018)
rf4 = try_clf(rf4, 'rf4')


# In[ ]:


use_clf(rf4,'csv/rf_50tree_100maxnode')


# In[ ]:


c = 'AB'


# In[ ]:


for a in reversed(c):
    print(a)


# In[ ]:


temp = pd.DataFrame({'a':['A','B','C','B'],
             'b':['DC','BD','BF','CC']})
temp


# In[ ]:


def op(col):
    return col.astype('category').cat.codes


# In[ ]:


# uniques = np.sort(pd.unique(temp.values.ravel()))
# temp.apply(lambda x: x.astype('category', categories=uniques))

le = LabelEncoder()
le.fit(temp.values.flat)

# Convert to digits.
temp = temp.apply(le.transform)
temp

