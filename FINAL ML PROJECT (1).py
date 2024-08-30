#!/usr/bin/env python
# coding: utf-8

# # Importing all the libraries

# In[3]:


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# In[4]:


df = pd.read_csv('Cervical.csv')


# In[5]:


df.replace('?', np.nan, inplace = True)


# In[6]:


imputer = KNNImputer(n_neighbors = 5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)


# In[7]:


df.isna()


# In[8]:


features = [
    'Age', 'Number of sexual partners', 'Num of pregnancies', 'Smokes', 
    'Hormonal Contraceptives', 'STDs (number)', 'STDs:cervical condylomatosis', 
    'STDs:vaginal condylomatosis', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B', 
    'Hinselmann', 'Schiller', 'Citology', 'Dx'
]


# In[9]:


X= df_imputed[features]
y = df_imputed['Biopsy']


# In[10]:


X = X.astype(float)


# In[11]:


y = y.astype(float)


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42, stratify = y)


# In[13]:


scaler = StandardScaler()


# In[14]:


model = LogisticRegression(max_iter =1000) 
rfe = RFE(model, n_features_to_select =10)


# In[15]:


smote = SMOTE(random_state=42) 


# In[16]:


pipeline = Pipeline(steps=[
    ('scaler', scaler),
    ('rfe', rfe),
    ('smote', smote),
    ('classifier', RandomForestClassifier(random_state=42, class_weight={0: 1, 1: 10}))  # Adjust class weights
])


# In[17]:


param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}


# In[18]:


grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1_macro')
grid_search.fit(X_train, y_train)


# In[19]:


best_model = grid_search.best_estimator_


# In[20]:


best_model


# In[21]:


y_pred = best_model.predict(X_test)


# In[22]:


y_pred


# In[23]:


print(confusion_matrix(y_test, y_pred))


# In[24]:


print("\nAccuracy Score:", accuracy_score(y_test, y_pred))


# In[25]:


print("\nClassification Report:\n", classification_report(y_test, y_pred))

