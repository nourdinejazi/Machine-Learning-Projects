#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split


# In[2]:


data=pd.read_csv(r'C:\Users\HP\Desktop\CarPrice_Assignment.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.symboling.unique()


# In[6]:


data.isnull().sum()


# In[7]:


import matplotlib.style as style
style.use('fivethirtyeight')
fig=plt.figure(constrained_layout=True,figsize=(12,8))
grid=gridspec.GridSpec(ncols=3,nrows=3,figure=fig)

ax1=fig.add_subplot(grid[0,:2])
ax1.set_title('Histogram')
sns.distplot(data.loc[:,'price'],norm_hist=True,ax=ax1)

ax2=fig.add_subplot(grid[1,:2])
ax1.set_title('QQ_plot')
stats.probplot(data.loc[:,'price'],plot=ax2);

ax3=fig.add_subplot(grid[:,2])
ax3.set_title('box plot')
sns.boxplot(data.loc[:,'price'],ax=ax3,orient='v')


# In[8]:


(data.corr())['price'].sort_values(ascending=False)[1:]


# In[9]:


def scatter(x,y,dataa):
    plt.figure(figsize=(8,4))
    sns.scatterplot(x,y,data=dataa);


# In[10]:


high_corr_fe=data.corr().index
high_corr_fe=high_corr_fe.drop('price')

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(20, 20))
k=0
for i in range(3) :
     for j in range(3):
          sns.scatterplot(high_corr_fe[k],'price',data=data,ax=axs[i][j])
          k=k+1
        


# In[11]:


data.price


# In[12]:


ds=data.copy()
X=data.drop('price',axis=1)
y=data.price
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=48)


# In[13]:


y_train=np.log1p(y_train)


# In[14]:


import matplotlib.style as style
style.use('fivethirtyeight')
fig=plt.figure(constrained_layout=True,figsize=(12,8))
grid=gridspec.GridSpec(ncols=3,nrows=3,figure=fig)

ax1=fig.add_subplot(grid[0,:2])
ax1.set_title('Histogram')
sns.distplot(y_train,norm_hist=True,ax=ax1)

ax2=fig.add_subplot(grid[1,:2])
ax1.set_title('QQ_plot')
stats.probplot(y_train,plot=ax2);

ax3=fig.add_subplot(grid[:,2])
ax3.set_title('box plot')
sns.boxplot(y_train,ax=ax3,orient='v')


# In[16]:


train=pd.concat([x_train,y_train],axis=1)
train.drop(['car_ID','CarName'],axis=1,inplace=True)
x_test.drop(['car_ID','CarName'],axis=1,inplace=True)


# In[17]:


plt.figure(figsize=(15,7))
mask=np.zeros_like(train.corr())
mask[np.triu_indices_from(mask)]=True
sns.heatmap(train.corr(),mask=mask,annot=True)
plt.title("Heatmap of all the Features", fontsize = 30);


# In[ ]:





# # feature engineering

# In[18]:


test=pd.concat([x_test,y_test],axis=1)
all_data=pd.concat([train,test],axis=0)
all_data.reset_index(drop=True,inplace=True)
X=all_data.drop(['price'], axis = 1)
all_data.drop(['price'], axis = 1, inplace = True)


# In[19]:


abs(train.corr())['price']<.2


# In[20]:


low_corr=['symboling','carheight','stroke','compressionratio','peakrpm']
all_data.drop(low_corr,axis=1,inplace=True)


# In[21]:


all_data['fuel_consumption']=all_data['citympg']+all_data['highwaympg']


# <h2>fixing skewness

# In[22]:


from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


# In[23]:


num_cols=all_data.dtypes[all_data.dtypes!='object'].index

skewed_feats=all_data[num_cols].apply(lambda x:skew(x)).sort_values(ascending=False)


# In[24]:


skewed_feats


# In[25]:


sns.distplot(all_data.horsepower,color='r')


# In[26]:


high_skew=skewed_feats[abs(skewed_feats>.5)].index
for i in high_skew :
  all_data[i]=boxcox1p(all_data[i],boxcox_normmax(all_data[i]+1))


# In[27]:


sns.distplot(all_data.horsepower,color='r')


# In[28]:


all_data.shape


# # <h1>Modeling the data

# In[29]:


X=all_data
y=pd.concat([y_train+y_test],axis=0).reset_index(drop=True)
x_train=all_data.iloc[:164,:]
x_test=all_data.iloc[164:,:]


# In[30]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score


# In[98]:


def cv_rmse(model, x,y):
    rmse = np.sqrt(-cross_val_score(model,x,y, scoring="neg_mean_squared_error", cv=5))
    return (rmse)


# In[63]:


cat_cols=all_data.dtypes[all_data.dtypes=='object'].index


# In[75]:


alpha=np.linspace(0.5,30,50)
pip=Pipeline([
    ('ct',ColumnTransformer([
        ('ss',StandardScaler(),num_cols),
        ('ohe',OneHotEncoder(),['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem'])
 ])) 
])


# In[76]:


lasso=LassoCV(alphas=alpha)
ridge=RidgeCV(alphas=alpha)
elasticnet=ElasticNetCV(alphas=alpha)


# In[82]:


x_train_tr=pip.fit_transform(x_train)


# In[109]:


lasso.fit(x_train_tr,y_train);
ridge.fit(x_train_tr,y_train);
elasticnet.fit(x_train_tr,y_train);


# In[112]:


print('lassocv rmse : ',cv_rmse(lasso,x_train_tr,y_train).mean())
print('ridgecv rmse : ',cv_rmse(ridge,x_train_tr,y_train).mean())
print('elasticnetcv rmse : ',cv_rmse(elasticnet,x_train_tr,y_train).mean())


# In[114]:


x_test_tr=pip.fit_transform(x_test)


# In[115]:


lasso.fit(x_test_tr,y_test);
ridge.fit(x_test_tr,y_test);
elasticnet.fit(x_test_tr,y_test);


# In[116]:


print('lassocv rmse test : ',cv_rmse(lasso,x_test_tr,y_test).mean())
print('ridgecv rmse test : ',cv_rmse(ridge,x_test_tr,y_test).mean())
print('elasticnetcv rmse test : ',cv_rmse(elasticnet,x_test_tr,y_test).mean())


# In[ ]:




