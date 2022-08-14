#!/usr/bin/env python
# coding: utf-8

# ### Importing/exploring the data  

# In[385]:


#importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[386]:


data = pd.read_csv('/Users/abhinavadarsh/Desktop/NEHA/WInter_2ndQuarter/ALY6015/W4/College.csv')
print(data.shape)


# In[387]:


# a few records of the data
data.head(5)


# In[388]:


#descriptive statistics
data.describe()    


# In[389]:


#checking null/missing values
print(data.isnull().sum())


# In[390]:


#creating dummies of the categorical variable
data.Private = pd.get_dummies(data[['Private']])


# In[391]:


#correlation between all the variables
data.corr()


# In[392]:


#correlation between variables
plt.figure(figsize=(10,8))
correlation_heatmap = sns.heatmap(data.corr())


# In[393]:


sns.set(rc = {'figure.figsize' : (12,8)})
sns.distplot(data['Grad_Rate'], bins = 20)


# ### Splitting the data into test and train sets 

# In[394]:


#predictors and target variables in X and y respectively
X = data[['Private', 'Apps', 'Accept', 'Enroll', 'Top10perc',
       'Top25perc', 'F_Undergrad', 'P_Undergrad', 'Outstate', 'Room_Board',
       'Books', 'Personal', 'PhD', 'Terminal', 'S_F_Ratio', 'perc_alumni',
       'Expend']]
y = data[['Grad_Rate']]


# In[395]:


from sklearn.model_selection  import train_test_split


# In[396]:


#splitting the data into 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)   


# ### Ridge Regression 

# #### Determining the performance of the fit model against the training set 

# In[397]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV         #we will use gridsearchcv for cross validation

ridge = Ridge()
parameters = {'alpha' : [1e-15, 1e-10, 1e-8, 1e-3, 1e-3, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring = 'neg_mean_squared_error', cv = 5)
ridge_model = ridge_regressor.fit(X_train, y_train)


# In[398]:


print(ridge_regressor.best_params_)   #this will give the most suitable lambda value
print(ridge_regressor.best_score_)   #neg mean squared error


# In[399]:


ridge_model = Ridge(alpha = 100).fit(X_train, y_train)          #Fitting ridge model on training data
coeffs = ridge_model.coef_                                          #Coefficients
coeffs


# In[400]:


y_pred3 = ridge_model.predict(X_test)       #predictions
y_pred3


# In[401]:


#Evaluation of the model

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred3)      #mean squared error
print('MSE :', mse)


# In[402]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred3))
print('RMSE', rmse)                #root mean squared error


# In[403]:


print('Rsquared: %.2f' % ridge_model.score(X_test, y_test))     #Rsquared value


# #### Determining the performance of the fit model against the test set

# In[113]:


#Determining the performance of the fit model against the test set

ridge_regressor1 = GridSearchCV(ridge, parameters, scoring = 'neg_mean_squared_error', cv = 5)
ridge_model1 = ridge_regressor1.fit(X_test, y_test)

print(ridge_regressor1.best_params_)   #this will give the most suitable lambda value
print(ridge_regressor1.best_score_)    #this will give the mean square error

ridge_model2 = Ridge(alpha = 100).fit(X_test, y_test)          #Fitting ridge model on test data
ridge_model2.coef_                                          #Coefficients


# ### Lasso Regression 

# #### Determining the performance of the fit model against the train set  

# In[404]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV         #we will use gridsearchcv for cross validation

#Determining the performance of the fit model against the train set
lasso = Lasso()
parameters2 = {'alpha' : [1e-15, 1e-10, 1e-8, 1e-3, 1e-3, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
lasso_regressor = GridSearchCV(lasso, parameters2, scoring = 'neg_mean_squared_error', cv = 5)
lasso_regressor.fit(X_train, y_train)


# In[405]:


print(lasso_regressor.best_params_)   #this will give the most suitable lambda value
print('neg_mean_squared_error :', lasso_regressor.best_score_)    #this will give the neg mean square error


# In[406]:


lasso_model = Lasso(alpha = 10).fit(X_train, y_train)          #Fitting Lasso model on training data
lasso_model.coef_                                          #Coefficients


# In[407]:


y1_pred = lasso_model.predict(X_test)       #predictions
y1_pred


# In[408]:


#Evaluation of the model

from sklearn.metrics import mean_squared_error

mse_lasso = mean_squared_error(y_test, y1_pred)      #mean squared error
print('MSE :', mse_lasso)


# In[409]:


rmse_lasso = np.sqrt(mean_squared_error(y_test, y1_pred))
print('RMSE', rmse_lasso)                #root mean squared error


# In[410]:


print('Rsquared: %.2f' % lasso_model.score(X_test, y_test))     #Rsquared value


# #### Determining the performance of the fit model against the test set

# In[ ]:



lasso_regressor1 = GridSearchCV(lasso, parameters2, scoring = 'neg_mean_squared_error', cv = 5)
lasso_regressor1.fit(X_test, y_test) 

print(lasso_regressor1.best_params_)   #this will give the most suitable lambda value
print(lasso_regressor1.best_score_)    #this will give the mean square error


# In[124]:



lasso_model2 = Lasso(alpha = 100).fit(X_test, y_test)          #Fitting Lasso model on test data
lasso_model2.coef_                                          #Coefficients


# ### Feature Selection 

# #### Forward selection

# In[71]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
sfs1 = SFS(lm, k_features=5, forward=True, verbose=2, scoring='neg_mean_squared_error')

sfs1 = sfs1.fit(X, y)


# In[72]:


sfs1.subsets_


# In[368]:


feat_names = list(sfs1.k_feature_names_)
print('Forward Selection Features :', feat_names)           #feature names


# #### Backward selection 

# In[75]:


lm = LinearRegression()
sfs2 = SFS(lm, k_features=5, forward=False, verbose=2, scoring='neg_mean_squared_error')

sfs2 = sfs2.fit(X, y)


# In[76]:


sfs2.subsets_


# In[369]:


feat_names = list(sfs2.k_feature_names_)
print('Backward Elimination Features :', feat_names)           #feature names


# #### Stepwise (bi directional) 

# In[78]:


lm3 = LinearRegression()
sfs3 = SFS(lm3, k_features=(4,5), forward=True, floating = True,verbose=2, scoring='neg_mean_squared_error')
sfs3.fit(X, y)


# In[79]:


sfs3.subsets_


# In[370]:


feat_names = list(sfs3.k_feature_names_)
print('Stepwise (bi-directional) Selection Features :', feat_names)           #feature names


# ### Multiple Linear Model fitting 

# In[411]:


X1 = data[['Apps', 'Top25perc', 'P_Undergrad', 'Outstate', 'perc_alumni']]
y1 = data[['Grad_Rate']]


# In[412]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.30, random_state = 0)


# In[413]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lm_regressor = LinearRegression()
lm_regressor.fit(X1_train, y1_train)


# In[414]:


y_pred2 = lm.predict(X1_train)
predictions = lm.predict(X1_test)
predictions


# In[372]:


mse = cross_val_score(lm_regressor, X1_train, y1_train, scoring = 'neg_mean_squared_error', cv = 5)   #cross validation on 5 experiments
mean_mse = np.mean(mse)    #mean squared error
print('neg_mean_squared_error :', mean_mse)


# In[264]:


#Model evaluation
# print the intercept
print(lm_regressor.intercept_)


# In[265]:


lm_regressor.coef_


# In[416]:


from sklearn.metrics import r2_score

print("R squared: {}".format(r2_score(y_true=y1_train,y_pred=y_pred2)))


# In[417]:


rmse_linear = np.sqrt(mean_squared_error(y1_test, predictions))
print('RMSE', rmse_linear)                #root mean squared error


# In[ ]:




