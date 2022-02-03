#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt


# In[2]:


trainfile = "data/train.csv"
testfile = "data/test.csv"
outputfile = "practice/output.txt"
weightfile = "practice/weight.txt"


# In[ ]:


gamma = 10**(-15)

def loss(W,X,y):
    y_pred = softmax(np.dot(X,W),axis=1)
    return -1*np.sum(np.multiply(y,np.log(np.clip(y_pred,gamma,1-gamma))))/X.shape[0]


# # A1

# In[7]:


def predictions(X,W):
        return (np.argmax(softmax(np.dot(X,W),axis=1),axis=1)+1).reshape(-1,1)

def part_a_1(trainfile, testfile, outputfile, weightfile, n0, it):
    losses = []
    times = []
    t1 = datetime.datetime.now()
    t_init = t1
    
    train = pd.read_csv(trainfile, index_col = 0)    
    test = pd.read_csv(testfile, index_col = 0)
    
    y_train = np.array(train['Length of Stay'])

    train = train.drop(columns = ['Length of Stay'])

    #Ensuring consistency of One-Hot Encoding

    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]
    
    y_train_1_hot = pd.get_dummies(y_train).to_numpy()
    
    n, m = X_train.shape
    n_test, m_test = X_test.shape
    
    X_train = np.append(np.ones([n,1]),X_train,axis=1)
    X_test = np.append(np.ones([n_test,1]),X_test,axis=1)

    m += 1
    m_test += 1
    
    W = np.zeros([m,8])
    
    for i in range(0,it):
        pred = softmax(np.dot(X_train,W),axis=1)
        W = W - n0/n*np.dot(X_train.T,np.subtract(pred,y_train_1_hot))
        t2 = datetime.datetime.now()
        times += [(t2-t1).seconds]
        losses += [loss(W,X_train,y_train_1_hot)]
        if((t2-t_init).seconds>=60):
            np.savetxt(weightfile,W.reshape(-1,1))
            np.savetxt(outputfile,predictions(X_test,W))
            t_init = datetime.datetime.now()
        
    np.savetxt(weightfile,W.reshape(-1,1))
    
    
    np.savetxt(outputfile,predictions(X_test,W))
    
    return (times,losses)


# In[19]:


times, losses = part_a_1(trainfile, testfile, outputfile, weightfile, 0.001, 100)


# In[20]:


plt.plot(times, losses)
plt.xlabel('Time (sec)')
plt.ylabel('Loss')
plt.title('A1 n0=0.001 it=100')
plt.show()


# In[8]:


times, losses = part_a_1(trainfile, testfile, outputfile, weightfile, 0.001, 300)


# In[9]:


plt.plot(times, losses)
plt.xlabel('Time (sec)')
plt.ylabel('Loss')
plt.title('A1 n0=0.001 it=300')
plt.show()


# In[10]:


times, losses = part_a_1(trainfile, testfile, outputfile, weightfile, 0.1, 300)


# In[11]:


plt.plot(times, losses)
plt.xlabel('Time (sec)')
plt.ylabel('Loss')
plt.title('A1 n0=0.01 it=300')
plt.show()


# In[ ]:





# # B3

# In[13]:


def predictions(X,W):
        return (np.argmax(softmax(np.dot(X,W),axis=1),axis=1)+1).reshape(-1,1)
    
def part_b_3(trainfile, testfile, outputfile, weightfile, n0, it, alpha, beta, batch):
    losses = []
    times = []
    t1 = datetime.datetime.now()
    t_init = t1
    
    train = pd.read_csv(trainfile, index_col = 0)    
    test = pd.read_csv(testfile, index_col = 0)
    
    y_train = np.array(train['Length of Stay'])

    train = train.drop(columns = ['Length of Stay'])

    #Ensuring consistency of One-Hot Encoding

    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]
    
    y_train_1_hot = pd.get_dummies(y_train).to_numpy()
    
    n, m = X_train.shape
    n_test, m_test = X_test.shape
    
    X_train = np.append(np.ones([n,1]),X_train,axis=1)
    X_test = np.append(np.ones([n_test,1]),X_test,axis=1)
    print(X_train.shape)

    m += 1
    m_test += 1
    
    W = np.zeros([m,8])
    
    for i in range(0,it):
        print()
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
        
        t2 = datetime.datetime.now()
        times += [(t2-t1).seconds]
        losses += [loss(W,X_train,y_train_1_hot)]
        if((t2-t_init).seconds>=60):
            np.savetxt(weightfile,W.reshape(-1,1))
            np.savetxt(outputfile,predictions(X_test,W))
            t_init = datetime.datetime.now()
        #pred = softmax(np.dot(X_train,W),axis=1)
        #while(loss()):
        #    n0 = beta*n0
        #W = W - n0/np.sqrt(i+1)/n*np.dot(X_train.T,np.subtract(pred,y_train_1_hot))
        
    np.savetxt(weightfile,W.reshape(-1,1))
    
    np.savetxt(outputfile,predictions(X_test,W))
    
    return (times,losses)


# In[14]:


times, losses = part_b_3(trainfile, testfile, outputfile, weightfile, 5, 300, 0.5, 0.7, 500)


# In[6]:


plt.plot(times, losses)
plt.xlabel('Time (sec)')
plt.ylabel('Loss')
plt.title('B3 n0=1 alpha=0.5 beta=0.7 it=300')
plt.show()


# In[16]:


plt.plot(times, losses)
plt.xlabel('Time (sec)')
plt.ylabel('Loss')
plt.title('B3 n0=1 alpha=0.5 beta=0.7 it=1000')
plt.show()


# In[ ]:




