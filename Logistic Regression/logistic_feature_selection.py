#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd



train_path = 'data/train_large.csv'
test_path = 'data/train.csv'

train = pd.read_csv(train_path, index_col = 0)    
test = pd.read_csv(test_path, index_col = 0)
    
y_train = np.array(train['Length of Stay'])

train = train.drop(columns = ['Length of Stay'])

y_test = np.array(test['Length of Stay'])

test = test.drop(columns = ['Length of Stay'])


# In[2]:


train.head()


# In[3]:


train.columns


# In[4]:


del train['Operating Certificate Number']
del train['Facility Id']
del train['Zip Code - 3 digits']
del train['Gender']
del train['Race']
del train['Ethnicity']
del train['CCS Diagnosis Code']
del train['CCS Procedure Code']
del train['APR DRG Code']
del train['APR Risk of Mortality']
del train['Birth Weight']

del test['Operating Certificate Number']
del test['Facility Id']
del test['Zip Code - 3 digits']
del test['Gender']
del test['Race']
del test['Ethnicity']
del test['CCS Diagnosis Code']
del test['CCS Procedure Code']
del test['APR DRG Code']
del test['APR Risk of Mortality']
del test['Birth Weight']


# In[5]:


train.columns


# In[6]:


#Ensuring consistency of One-Hot Encoding

data = pd.concat([train, test], ignore_index = True)
cols = train.columns
cols = cols[:-1]
data = pd.get_dummies(data, columns=cols, drop_first=True)
data = data.to_numpy()
X_train = data[:train.shape[0], :]
X_test = data[train.shape[0]:, :]


# In[7]:


print(X_train)
print(X_train.shape)


# In[ ]:





# In[8]:


from scipy.special import softmax


# In[9]:


y_train_1_hot = pd.get_dummies(y_train).to_numpy()
print(y_train_1_hot)


# In[10]:


print(X_train.shape)
print(X_test.shape)


# In[11]:


n, m = X_train.shape
n_test, m_test = X_test.shape


# In[ ]:





# In[12]:


X_train = np.append(np.ones([n,1]),X_train,axis=1)
X_test = np.append(np.ones([n_test,1]),X_test,axis=1)

m += 1
m_test += 1


# In[13]:


print(X_train.shape)
print(X_test.shape)


# In[14]:


W = np.zeros([m,8])


# In[ ]:





# In[15]:


gamma = 10**(-15)

def loss(W,X,y):
    y_pred = softmax(np.dot(X,W),axis=1)
    return -1*np.sum(np.multiply(y,np.log(np.clip(y_pred,gamma,1-gamma))))/X.shape[0]


# In[16]:


n0 = 5
it = 250
alpha = 0.5
beta = 0.7
batch = 300


# In[ ]:





# In[ ]:





# In[17]:


for i in range(0,it):
    print(i)
    pred = softmax(np.dot(X_train,W),axis=1)
    grad = np.dot(X_train.T,np.subtract(pred,y_train_1_hot))/n
    summ = alpha*np.sum((grad)**2)
    W_next = W - n0*grad
    while(loss(W,X_train,y_train_1_hot)-loss(W_next,X_train,y_train_1_hot)<n0*summ):
        #print(loss(W,X_train,y_train_1_hot)-loss(W_next,X_train,y_train_1_hot))
        #print(n0*summ)
        n0 = beta*n0
        W_next = W - n0*grad
    curr_batch = batch
    while(curr_batch<=n):
        X_train_curr = X_train[curr_batch-batch:curr_batch]
        y_train_1_hot_curr = y_train_1_hot[curr_batch-batch:curr_batch]
        pred = softmax(np.dot(X_train_curr,W),axis=1)
        W = W - n0/batch*np.dot(X_train_curr.T,np.subtract(pred,y_train_1_hot_curr))
        curr_batch += batch
    #pred = softmax(np.dot(X_train,W),axis=1)
    #while(loss()):
    #    n0 = beta*n0
    #W = W - n0/np.sqrt(i+1)/n*np.dot(X_train.T,np.subtract(pred,y_train_1_hot))


# In[18]:


print(W.shape)


# In[19]:


print(np.array(W).reshape(-1,1))


# In[20]:


print(y_test.reshape(-1,1))


# In[21]:


def predictions(X,W):
     return (np.argmax(softmax(np.dot(X,W),axis=1),axis=1)+1).reshape(-1,1)


# In[22]:


print(predictions(X_test,W))


# In[23]:


sum((y_test.reshape(-1,1)) == (predictions(X_test,W)))/len((y_test.reshape(-1,1)) == (predictions(X_test,W)))


# In[ ]:




