#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


df = pd.read_csv("Assignment_1/data/train_large.csv",index_col=[0])
df.head()


# In[3]:


df3 = pd.read_csv("Assignment_1/data/train.csv",index_col=[0])
df3.head()


# In[ ]:





# In[4]:


unique_columns = df.nunique(axis=0)
print(unique_columns)


# In[5]:


#del df["Health Service Area"]
#del df["Hospital County"]
del df["Operating Certificate Number"]
del df["Facility Id"]
#del df["Facility Name"]
del df["Zip Code - 3 digits"]
del df["CCS Diagnosis Code"]
del df["CCS Diagnosis Description"]
del df["CCS Procedure Code"]
del df["CCS Procedure Description"]
del df["APR DRG Code"]
del df["APR DRG Description"]
del df['Birth Weight']

del df["APR MDC Code"]
del df["APR Severity of Illness Code"]
del df["APR Risk of Mortality"]
#del df["APR Medical Surgical Description"]


#del df["Age Group"]
del df["Gender"]
del df["Race"]
del df["Ethnicity"]
#del df["Type of Admission"]
#del df["Patient Disposition"]
#del df["APR MDC Description"]
#del df["APR Severity of Illness Description"]
#del df["Payment Typology 1"]
#del df["Payment Typology 2"]
#del df["Payment Typology 3"]
#del df["Emergency Department Indicator"]


# In[6]:


del df3["Operating Certificate Number"]
del df3["Facility Id"]
#del df["Facility Name"]
del df3["Zip Code - 3 digits"]
del df3["CCS Diagnosis Code"]
del df3["CCS Diagnosis Description"]
del df3["CCS Procedure Code"]
del df3["CCS Procedure Description"]
del df3["APR DRG Code"]
del df3["APR DRG Description"]
del df3['Birth Weight']

del df3["APR MDC Code"]
del df3["APR Severity of Illness Code"]
del df3["APR Risk of Mortality"]
#del df["APR Medical Surgical Description"]


#del df["Age Group"]
del df3["Gender"]
del df3["Race"]
del df3["Ethnicity"]


# # New Dataframe

# In[7]:


df.head()


# In[8]:


unique_columns = df.nunique(axis=0)
print(unique_columns)


# # Performance Metrics

# In[9]:


def r_2_score(y_pred,y_test):
    mean_target = np.mean(y_test)
    num = np.sum((y_test-y_pred)**2)
    denom = np.sum((y_test-mean_target)**2)
    return 1-(num/denom)


# In[10]:


def loss(y_pred,y_test):
    diff = np.subtract(y_pred,y_test)
    return np.dot(diff.T,diff)/np.dot(y_test.T,y_test)


# In[11]:


def performance(X,y,X_test,y_test,lam_da):
    w = np.dot(np.linalg.inv(np.add(np.dot(X.T,X),lam_da*np.identity(X.shape[1]))),np.dot(X.T,y))
    w.shape = (w.shape[0],1)
    y_pred = np.dot(X_test,w)
    return loss(y_pred,y_test)


# In[12]:


def lasso_performance(X,y,X_test,y_test,alpha):
    reg = linear_model.LassoLars(alpha=alpha)
    reg.fit(X, y)
    return reg.score(X_test, y_test)


# In[13]:


def r_2_performance(X,y,X_test,y_test,lam_da):
    w = np.dot(np.linalg.inv(np.add(np.dot(X.T,X),lam_da*np.identity(X.shape[1]))),np.dot(X.T,y))
    w.shape = (w.shape[0],1)
    y_pred = np.dot(X_test,w)
    return r_2_score(y_pred,y_test)


# # Create Matrix

# In[14]:


X = df.to_numpy()
print(X.shape)


# In[15]:


np.random.shuffle(X)
y = X[:,-1]
X = X[:,:-1]
n,m = X.shape


# In[ ]:





# In[16]:


X_t = df3.to_numpy()
print(X_t.shape)
y_t = X_t[:,-1]
X_t = X_t[:,:-1]
n_t,m = X_t.shape


# In[71]:


y_t = y_t.reshape(X_t.shape[0],1)


# # Transform Data

# In[17]:


lengths_of_stay = X[:,df.columns.get_loc('Length of Stay')].reshape(X.shape[0],1)


# In[18]:


lengths_of_stay_t = X_t[:,df3.columns.get_loc('Length of Stay')].reshape(X_t.shape[0],1)


# In[19]:


unique_columns = df.nunique(axis=0)
print(unique_columns)


# In[20]:


len_of_1_hot = 0


# In[21]:


X_columns = np.array(list(df.columns))[:-1]
print(X_columns)


# In[22]:


hot_columns = []


# In[23]:


col_name = 'Health Service Area'
length = 8
col_num = df.columns.get_loc(col_name)

len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[24]:


col_name = 'Health Service Area'
length = 8
col_num = df3.columns.get_loc(col_name)


print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)

del df3[col_name]


# In[ ]:





# In[25]:


col_name = 'Hospital County'
length = 57
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[26]:


col_name = 'Hospital County'
length = 57
col_num = df3.columns.get_loc(col_name)



print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)


del df3[col_name]


# In[ ]:





# In[27]:


col_name = 'Facility Name'
length = 212
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[28]:


col_name = 'Facility Name'
length = 212
col_num = df3.columns.get_loc(col_name)



print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)


del df3[col_name]


# In[ ]:





# In[29]:


col_name = 'Age Group'
length = 5
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[30]:


col_name = 'Age Group'
length = 5
col_num = df3.columns.get_loc(col_name)


print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)


del df3[col_name]


# In[ ]:





# In[31]:


col_name = 'Type of Admission'
length = 6
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[32]:


col_name = 'Type of Admission'
length = 6
col_num = df3.columns.get_loc(col_name)


print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)


del df3[col_name]


# In[ ]:





# In[33]:


col_name = 'Patient Disposition'
length =19
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[34]:


col_name = 'Patient Disposition'
length =19
col_num = df3.columns.get_loc(col_name)



print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)

del df3[col_name]


# In[ ]:





# In[35]:


col_name = 'APR MDC Description'
length =24
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[36]:


col_name = 'APR MDC Description'
length =24
col_num = df3.columns.get_loc(col_name)



print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)

del df3[col_name]


# In[ ]:





# In[37]:


col_name = 'APR Severity of Illness Description'
length = 4
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[38]:


col_name = 'APR Severity of Illness Description'
length = 4
col_num = df3.columns.get_loc(col_name)



print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)


del df3[col_name]


# In[ ]:





# In[39]:


col_name = 'APR Medical Surgical Description'
length = 2
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[40]:


col_name = 'APR Medical Surgical Description'
length = 2
col_num = df3.columns.get_loc(col_name)


print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)


del df3[col_name]


# In[ ]:





# In[41]:


col_name = 'Payment Typology 1'
length = 10
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[42]:


col_name = 'Payment Typology 1'
length = 10
col_num = df3.columns.get_loc(col_name)


print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)


del df3[col_name]


# In[ ]:





# In[43]:


col_name = 'Payment Typology 2'
length = 11
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[44]:


col_name = 'Payment Typology 2'
length = 11
col_num = df3.columns.get_loc(col_name)


print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)

del df3[col_name]


# In[ ]:





# In[45]:


col_name = 'Payment Typology 3'
length = 11
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[76]:


col = [5,0, 2, 4, 1]
print(col)
new_cols = [[1],[2],[3],[4],[5]]*np.eye(6)[col]
print(new_cols)


# In[ ]:





# In[ ]:





# In[46]:


col_name = 'Payment Typology 3'
length = 11
col_num = df3.columns.get_loc(col_name)



print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)


del df3[col_name]


# In[ ]:





# In[47]:


col_name = 'Emergency Department Indicator'
length = 2
col_num = df.columns.get_loc(col_name)


len_of_1_hot += length

print(X.shape)
col = X[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay*np.eye(length)[col]
X = np.delete(X,col_num,1)
X = np.append(X,new_cols,axis=1)
print(X.shape)
del df[X_columns[col_num]]
X_columns = np.delete(X_columns, col_num)

for i in range(1, length+1):
    hot_columns += [col_name+str(i)]


# In[ ]:





# In[48]:


col_name = 'Emergency Department Indicator'
length = 2
col_num = df3.columns.get_loc(col_name)


print(X_t.shape)
col = X_t[:,col_num].astype(int) - 1
print(col)
col[col>=length] = length-1
col[col<0] = 0
print(col)
new_cols = lengths_of_stay_t*np.eye(length)[col]
X_t = np.delete(X_t,col_num,1)
X_t = np.append(X_t,new_cols,axis=1)
print(X_t.shape)


del df3[col_name]


# In[ ]:





# In[49]:


unique_columns = df.nunique(axis=0)
print(unique_columns)


# In[50]:


print(len_of_1_hot)


# In[51]:


hot_columns = np.array(hot_columns)
X_columns = np.append(X_columns,hot_columns)


# In[52]:


print(X_columns)


# In[53]:


print(X.shape)
print(len(X_columns))


# # Split

# In[54]:


n_test_1 = n//10
n_test_2 = n_test_1
n_train = n - n_test_1 - n_test_2
X_train = X[:n_train]
y_train = y[:n_train]
X_test_1 = X[n_train:n_train+n_test_1]
y_test_1 = y[n_train:n_train+n_test_1]
X_test_2 = X[n_train+n_test_1:]
y_test_2 = y[n_train+n_test_1:]


# In[55]:


y_train = y_train.reshape(n_train,1)
y_test_1 = y_test_1.reshape(n_test_1,1)
y_test_2 = y_test_2.reshape(n_test_2,1)


# In[56]:


print(X_train.shape)
y_train.shape


# # LassoLars
# 

# In[57]:


alpha = 0.001


# In[58]:


from sklearn import linear_model
reg = linear_model.LassoLars(alpha=alpha)


# In[59]:


reg.fit(X_train, y_train)


# In[60]:


reg.score(X_train, y_train)


# In[62]:


reg.score(X_t, y_t)


# In[59]:


print(reg.coef_)


# In[55]:


print(len(reg.coef_))
print(len(reg.coef_[reg.coef_!=0]))


# In[64]:


print(X_columns[reg.coef_!=0])


# In[ ]:





# # Remove Useless Columns

# In[65]:


X_train = X_train[:,reg.coef_!=0]
X_columns = X_columns[reg.coef_!=0]
print(X_train.shape)
print(X_columns.shape)
X_test_1 = X_test_1[:,reg.coef_!=0]
X_test_2 = X_test_2[:,reg.coef_!=0]


# In[ ]:





# # Ridge Regression

# In[66]:


print(performance(X_train,y_train,X_test_1,y_test_1,0.001))


# In[72]:


print(r_2_performance(X_train,y_train,X_t,y_t,0.001))


# In[ ]:





# In[135]:


print(X_train.shape)
print(X_t.shape)


# In[ ]:





# In[ ]:




