
# coding: utf-8

# # Import the required modules

# In[66]:


import os
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[2]:


os.chdir("E:\Data Science\Capstone Projects\Capstone Project_Xg boost")


# # Read the dataset

# In[3]:


dataset=pd.read_csv("Purchase.csv")


# In[4]:


#dataset.head(2)
dataset.Field6.dtypes


# # Extract month, weekday,year from Original_Quote_Date column using date_time function in Pandas

# In[5]:


dataset['Date']=pd.to_datetime(pd.Series(dataset['Original_Quote_Date']))


# In[7]:


dataset=dataset.drop(["Original_Quote_Date" ], axis=1)
dataset.head(2)


# In[8]:


dataset['weekday']=dataset['Date'].dt.dayofweek
dataset['weekyear']=dataset['Date'].dt.weekofyear
dataset['Year']=dataset['Date'].dt.year
dataset['Month']=dataset['Date'].dt.month


# # Write a simple for loop to get the columns with datatypes as object and fill those columns missing values with Place Holder value.

# In[9]:


cols=[]
for i in list(dataset.columns):
    if dataset[i].dtypes == 'O':
        cols.append(i)
        
    
    


# In[10]:


cols


# In[11]:


dataset[cols]=dataset[cols].fillna(-1)


# In[14]:


dataset[cols].isnull().sum().sum()


# In[16]:


dataset.isnull().sum().sum()


# # Convert the Object columns to Numerical using Label Encoder(Not Dummies)

# In[23]:


lbl=LabelEncoder()
for cols in dataset:
    dataset[cols]=lbl.fit_transform(list(dataset[cols].values))
    


# # Train Test Split

# In[28]:


y=dataset[['QuoteConversion_Flag']]


# In[30]:


X=dataset.drop(["QuoteConversion_Flag"],axis=1)


# In[33]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)


# # Build the Model now

# In[36]:


import xgboost
from xgboost import XGBClassifier


# In[77]:


eval_set = [(X_test, y_test)]
model=xgb.XGBClassifier(n_estimators=100,eta=0.1,objective="binary:logistic",silent=0,eval_metric="error",
                        eval_set=eval_set,verbose=10,max_depth=10,
                        subsample=0.6,colsample_bytree=0.6,reg_lambda=0.6,gamma=5)


# In[78]:


model.fit(X_train,y_train)


# In[63]:


preds =model.predict(X_test)


# # best params:eta=0.1,n_estimators=100,max_depth=default(3),f1:0.74
# # best params:eta=0.1,n_estimators=100,max_depth=10,f1:0.77
# # bestparams:eta=0.1,n_estimators=100,max_depth=10,subsample=0.6,colsample_bytree=0.6,f1:0.78

# In[64]:


print(classification_report(y_test,preds))


# In[65]:


accuracy_score(preds,y_test)


# # Applying KFold and cross_val_scores

# In[68]:


kfold=StratifiedKFold(n_splits=3,random_state=0)
result=cross_val_score(model,X_train,y_train,cv=kfold)


# In[75]:


print(result.std())

print(result)

