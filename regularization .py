#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from seaborn import load_dataset
data=load_dataset('mpg')


# In[3]:


data


# In[4]:


tr,te=train_test_split(data,test_size=.25,random_state=83)


# In[5]:


tr


# In[6]:


te


# In[7]:


models={}


# In[8]:


quantitative_features=tr.dtypes[tr.dtypes!='object'].index
quantitative_features=quantitative_features.drop(quantitative_features[0])
quantitative_features


# In[9]:


for i in range(len(quantitative_features)) :
    features=quantitative_features[:(i+1)]
    name=','.join(name[0] for name in features)
    
    model=Pipeline([
        ('sc',ColumnTransformer([
            ('keep','passthrough',features),
        ])),
        ('im',SimpleImputer()),
        ('lm',LinearRegression())
    ])
    model.fit(tr,tr['mpg']);
    
    models[name]=model


# In[10]:


models.keys()


# In[11]:


from sklearn.model_selection import cross_val_score
 


# In[12]:


def rmse(model,x,y):
    return np.sqrt(np.mean((y-model.predict(x))**2))


# In[13]:


np.mean(cross_val_score(models['c'],tr,tr['mpg'],scoring=rmse,cv=5))


# In[41]:


def compare_models (models):
    sns.set_style('whitegrid')
    traing_rmse=[rmse(model,tr,tr['mpg']) for model in models.values()]
    cv_rmse=[np.mean(cross_val_score(model,tr,tr['mpg'],scoring=rmse,cv=5)) for model in models.values() ]
    name=list(models.keys())

    plt.figure(figsize=(10,5))
    w=0.4
    bar1=np.arange(len(name))
    bar2=[i+w for i in bar1]
    plt.bar(bar1,traing_rmse,w,label='training_rmse')
    plt.bar(bar2,cv_rmse,w,label='cv_rmse')
    plt.xticks(bar1+w/2,name,rotation=70)
    plt.legend()


# In[15]:


from sklearn.preprocessing import OneHotEncoder


# In[16]:


model=Pipeline([
    ('sc',ColumnTransformer([
        ('keep','passthrough',quantitative_features),
        ('ohe',OneHotEncoder(),['origin'])
    ])),
    ('si',SimpleImputer()),
    ('model',LinearRegression())
])


# In[17]:


model.fit(tr,tr['mpg'])
name=','.join(name[0] for name in quantitative_features)+',o'
models[name]=model


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer
model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", "passthrough", quantitative_features),
        ("origin_encoder", OneHotEncoder(), ["origin"]),
        ("text", CountVectorizer(), "name")
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", LinearRegression())
])


# In[19]:


model.fit(tr, tr['mpg'])
name = ",".join([name[0] for name in quantitative_features]) + ",o,n"
models[name] = model


# In[20]:


compare_models(models)


# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV


# In[23]:


ridge_model=Pipeline([
    ('ct',ColumnTransformer([
        ('ss',StandardScaler(),quantitative_features),
        ('ohe',OneHotEncoder(),['origin']),
        ("text", CountVectorizer(), "name")
    ])),
    ('si',SimpleImputer()),
    ('model',Ridge())
])


# In[24]:


alphas = np.linspace(0.5,40,50)
cv_values=[]
train_values=[]
test_values=[]
for alpha in alphas : 
    ridge_model.set_params(model__alpha=alpha)
    cv_values.append(np.mean(cross_val_score(ridge_model,tr,tr['mpg'],scoring=rmse,cv=5)))
    ridge_model.fit(tr,tr['mpg'])
    train_values.append(rmse(ridge_model,tr,tr['mpg']))
    test_values.append(rmse(ridge_model,te,te['mpg']))
    


# In[25]:



plt.figure(figsize=(15,10))
plt.scatter(x=alphas,y=cv_values,label='cv_values',c='red', 
    marker='o')
plt.plot(alphas,cv_values,c='red')



plt.scatter(x=alphas,y=train_values,label='train_values',c='blue')
plt.plot(alphas,train_values,c='b')


plt.scatter(x=alphas,y=test_values,label='test_values',c='green', )
plt.plot(alphas,test_values,c='g')

plt.legend()


# In[26]:


best_alpha=alphas[np.argmin(cv_values)]
best_alpha
ridge_model.set_params(model__alpha=best_alpha)
ridge_model.fit(tr,tr['mpg'])
models['ridgenbestalpha']=ridge_model
compare_models(models)


# In[27]:


alphas = np.linspace(0.5, 3, 30)

ridge_model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", StandardScaler(), quantitative_features),
        ("origin_encoder", OneHotEncoder(), ["origin"]),
        ("text", CountVectorizer(), "name")
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", RidgeCV(alphas=alphas))
])


# In[28]:


ridge_model.fit(tr, tr['mpg'])
models["RidgeCV"] = ridge_model
compare_models(models)


# In[29]:


from sklearn.linear_model import Lasso, LassoCV


# In[39]:


lasso_model = Pipeline([
    ("SelectColumns", ColumnTransformer([
        ("keep", StandardScaler(), quantitative_features),
        ("origin_encoder", OneHotEncoder(), ["origin"]),
        ("text", CountVectorizer(), "name")
    ])),
    ("Imputation", SimpleImputer()),
    ("LinearModel", LassoCV())
    
])


# In[42]:


lasso_model.fit(tr, tr['mpg'])
models["LassoCV"] = lasso_model
compare_models(models)


# In[ ]:




