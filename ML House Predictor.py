#!/usr/bin/env python
# coding: utf-8

# # ML House Predictor

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
# To use plt.show in Jupyter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit  
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
#Model
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor() # To Change Model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump,load


# In[2]:


housing = pd.read_csv("house_project_data.csv")
# housing = housing.iloc[:, :-4]/ Coloumns


# In[3]:


housing[:5]


# In[4]:


housing.describe()


# # HISTOGRAM PLOTTING

# In[5]:


# housing.hist(bins=50, figsize=(20,10))
# plt.show()


# # Train - Test Splitting ->Function

# In[6]:


# def split_train_test(data,test_ratio):
#     np.random.seed(5) #--> To FIX the Shuffling value x can be anything --> it is the order
#     shuffled = np.random.permutation(len(data)) #--> random Shuffling all the data indices
# #     print(shuffled)
#     test_set_size= int(len(data)*test_ratio)
#     test_indices= shuffled[:test_set_size]
#     train_indices= shuffled[test_set_size:]
#     return data.iloc[train_indices] , data.iloc[test_indices]


# In[7]:


# train_set, test_set= split_train_test(housing,0.2)


# In[8]:


# print(f"Rows in Train Set : {len(train_set)}\n Rows in Test Set : {len(test_set)} ")


# In[9]:


train_set, test_set= train_test_split(housing, test_size=0.2,random_state=42)
print(f"Rows in Train Set : {len(train_set)}\n Rows in Test Set : {len(test_set)} ")


# In[10]:


split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing, housing[' CHAS']): #For equal ratio of CHAS
    strat_train_set = housing.loc[train_index]
    strat_test_set= housing.loc[test_index]


# In[11]:


strat_train_set #strat_train_set['CHAS'].value_counts()


# # Looking for Corelations(Value depending on others)  
# 

# In[12]:


corr_matrix=housing.corr()
corr_matrix[' MEDV'].sort_values(ascending = False)


# In[13]:


attributes = [" MEDV", " RM"]
scatter_matrix(housing[attributes], figsize= (12,8))
# housing.plot(kind = "scatter",x=" RM",y=" MEDV",alpha=0.6)


# # Seprating Features & Labels

# In[14]:


housing = strat_train_set.drop(" MEDV",axis=1)
housing_labels= strat_train_set[" MEDV"].copy()


# # Imputer to fill Median Values in Missing Coloumns

# In[15]:


imputer =  SimpleImputer(strategy="median")
imputer.fit(housing)
imputer.statistics_


# In[16]:


X= imputer.transform(housing)
housing_tr= pd.DataFrame(X,columns=housing.columns)
# housing_tr.describe()  


# # Feature Scaling
# Primarily , two types of feature scaling methods:
# 1. Min-Max scaling (Normalization)
# (value - min)/(max-min)
# sklearn provides a class called *MinMaxScaler* for this
# 
# 2. Standardization
# (value - mean)/std
# Sklearn provides a class called StandardScaler for this

# # Creating Pipeline
# To make changes in code/Model easily or to automate code

# In[17]:


my_pipeline  = Pipeline([
                        ('imputer',SimpleImputer(strategy = "median")),
                       #ADDING AS MANY WANT.....
                       ('std_scaler', StandardScaler())
                       ])


# In[18]:


housing_num_tr = my_pipeline.fit_transform(housing) #Training & Transforming FIT -> Stores Mean & STD , TRANSFORM -> APPLIES IT |
housing_num_tr # In test we use only transform with previous mean & std


# In[19]:


housing_num_tr.shape


# # Selecting Model

# In[20]:


# model = LinearRegression()
model.fit(housing_num_tr,housing_labels) # Learn about Data(Mean & Scale) & Transform - Uses learned data in data transformation


# In[33]:


some_data = housing.iloc[:5]
some_labels= housing_labels.iloc[:5]
prepared_data= my_pipeline.transform(some_data)
model.predict(prepared_data) # SEE PREDICTION


# In[22]:


list(some_labels)


# # Evaluating the Model

# In[23]:


housing_predictions= model.predict(housing_num_tr)
lin_mse= mean_squared_error(housing_labels,housing_predictions)
lin_rmse= np.sqrt(lin_mse)
lin_mse


# # Using better eval. Technique - Cross validation
# 1 2 3 4 5 6 7 8 9 10

# In[24]:


scores= cross_val_score(model, housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores


# In[25]:


def print_scores(scores):
       print("Scores : ",scores)
       print("Mean : ",scores.mean())
       print("Standard Deviation : ",scores.std())


# In[26]:


print_scores(rmse_scores)


# # Saving The Model

# In[27]:


dump(model, 'House Pred.joblib')


# # Testing The Model

# In[28]:


X_test = strat_test_set.drop(" MEDV",axis=1)
Y_test = strat_test_set[" MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions= model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions,list(Y_test))


# In[29]:


final_rmse


# In[30]:


prepared_data[0]


# # Using The Model

# In[31]:


from joblib import dump,load
import numpy as np
model = load('House Pred.joblib')

features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




