import pandas as pd
import numpy as np
from scipy.special import softmax
import sys
import datetime

gamma = 10**(-15)

def loss(W,X,y):
    y_pred = softmax(np.dot(X,W),axis=1)
    return -1*np.sum(np.multiply(y,np.log(np.clip(y_pred,gamma,1-gamma))))/X.shape[0]

def part_a_1(trainfile, testfile, outputfile, weightfile, n0, it):
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
        
    np.savetxt(weightfile,W.reshape(-1,1))
    
    def predictions(X,W):
        return (np.argmax(softmax(np.dot(X,W),axis=1),axis=1)+1).reshape(-1,1)
    
    np.savetxt(outputfile,predictions(X_test,W))
    

    
def part_a_2(trainfile, testfile, outputfile, weightfile, n0, it):
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
        W = W - n0/n/np.sqrt(i+1)*np.dot(X_train.T,np.subtract(pred,y_train_1_hot))
        
    np.savetxt(weightfile,W.reshape(-1,1))
    
    def predictions(X,W):
        return (np.argmax(softmax(np.dot(X,W),axis=1),axis=1)+1).reshape(-1,1)
    
    np.savetxt(outputfile,predictions(X_test,W))
    
    
def part_a_3(trainfile, testfile, outputfile, weightfile, n0, it, alpha, beta):
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

    n0_old = n0
    
    for i in range(0,it):
        n0 = n0_old
        pred = softmax(np.dot(X_train,W),axis=1)
        grad = np.dot(X_train.T,np.subtract(pred,y_train_1_hot))/n
        summ = alpha*np.sum((grad)**2)
        W_next = W - n0*grad
        while(loss(W,X_train,y_train_1_hot)-loss(W_next,X_train,y_train_1_hot)<n0*summ):
            #print(loss(W,X_train,y_train_1_hot)-loss(W_next,X_train,y_train_1_hot))
            #print(n0*summ)
            n0 = beta*n0
            W_next = W - n0*grad
        W = W_next
        #pred = softmax(np.dot(X_train,W),axis=1)
        #while(loss()):
        #    n0 = beta*n0
        #W = W - n0/np.sqrt(i+1)/n*np.dot(X_train.T,np.subtract(pred,y_train_1_hot))
    
    np.savetxt(weightfile,W.reshape(-1,1))
    
    def predictions(X,W):
        return (np.argmax(softmax(np.dot(X,W),axis=1),axis=1)+1).reshape(-1,1)
    
    np.savetxt(outputfile,predictions(X_test,W))
    
def part_a(trainfile, testfile, outputfile, weightfile, param):
    params = open(param, "r")
    lines = params.readlines()
    if(int(lines[0])==1):
        part_a_1(trainfile, testfile, outputfile, weightfile, float(lines[1]), int(lines[2]))
    elif(int(lines[0])==2):
        part_a_2(trainfile, testfile, outputfile, weightfile, float(lines[1]), int(lines[2]))        
    else:
        n0, alpha, beta = lines[1].split(",")
        part_a_3(trainfile, testfile, outputfile, weightfile, float(n0), int(lines[2]), float(alpha), float(beta))


        
def part_b_1(trainfile, testfile, outputfile, weightfile, n0, it, batch):
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
        curr_batch = batch
        while(curr_batch<=n):
            X_train_curr = X_train[curr_batch-batch:curr_batch]
            y_train_1_hot_curr = y_train_1_hot[curr_batch-batch:curr_batch]
            pred = softmax(np.dot(X_train_curr,W),axis=1)
            W = W - n0/batch*np.dot(X_train_curr.T,np.subtract(pred,y_train_1_hot_curr))
            curr_batch += batch
        
    np.savetxt(weightfile,W.reshape(-1,1))
    
    def predictions(X,W):
        return (np.argmax(softmax(np.dot(X,W),axis=1),axis=1)+1).reshape(-1,1)
    
    np.savetxt(outputfile,predictions(X_test,W))
    
def part_b_2(trainfile, testfile, outputfile, weightfile, n0, it, batch):
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
        curr_batch = batch
        while(curr_batch<=n):
            X_train_curr = X_train[curr_batch-batch:curr_batch]
            y_train_1_hot_curr = y_train_1_hot[curr_batch-batch:curr_batch]
            pred = softmax(np.dot(X_train_curr,W),axis=1)
            W = W - n0/np.sqrt(i+1)/batch*np.dot(X_train_curr.T,np.subtract(pred,y_train_1_hot_curr))
            curr_batch += batch
        
    np.savetxt(weightfile,W.reshape(-1,1))
    
    def predictions(X,W):
        return (np.argmax(softmax(np.dot(X,W),axis=1),axis=1)+1).reshape(-1,1)
    
    np.savetxt(outputfile,predictions(X_test,W))
    
def part_b_3(trainfile, testfile, outputfile, weightfile, n0, it, alpha, beta, batch):
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

    n0_old = n0
    
    for i in range(0,it):
        n0 = n0_old
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
        
    np.savetxt(weightfile,W.reshape(-1,1))
    
    def predictions(X,W):
        return (np.argmax(softmax(np.dot(X,W),axis=1),axis=1)+1).reshape(-1,1)
    
    np.savetxt(outputfile,predictions(X_test,W))
    
def part_b(trainfile, testfile, outputfile, weightfile, param):
    params = open(param, "r")
    lines = params.readlines()
    if(int(lines[0])==1):
        part_b_1(trainfile, testfile, outputfile, weightfile, float(lines[1]), int(lines[2]), int(lines[3]))
    elif(int(lines[0])==2):
        part_b_2(trainfile, testfile, outputfile, weightfile, float(lines[1]), int(lines[2]), int(lines[3]))        
    else:
        n0, alpha, beta = lines[1].split(",")
        part_b_3(trainfile, testfile, outputfile, weightfile, float(n0), int(lines[2]), float(alpha), float(beta), int(lines[3]))
                 
                 
def part_c(trainfile, testfile, outputfile, weightfile):
    def predictions(X,W):
        return (np.argmax(softmax(np.dot(X,W),axis=1),axis=1)+1).reshape(-1,1)
    
    n0 = 5
    alpha = 0.5
    beta = 0.7
    batch = 500
    it = 100000
    
    t_init = datetime.datetime.now()
    
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
            if((t2-t_init).seconds>=30):
                np.savetxt(weightfile,W.reshape(-1,1))
                np.savetxt(outputfile,predictions(X_test,W))
                t_init = datetime.datetime.now()
        #pred = softmax(np.dot(X_train,W),axis=1)
        #while(loss()):
        #    n0 = beta*n0
        #W = W - n0/np.sqrt(i+1)/n*np.dot(X_train.T,np.subtract(pred,y_train_1_hot))
        
    np.savetxt(weightfile,W.reshape(-1,1))
    
    np.savetxt(outputfile,predictions(X_test,W))
    
def part_d(trainfile, testfile, outputfile, weightfile):
    def predictions(X,W):
        return (np.argmax(softmax(np.dot(X,W),axis=1),axis=1)+1).reshape(-1,1)
    
    n0 = 5
    alpha = 0.5
    beta = 0.7
    batch = 500
    it = 100000
    
    t_init = datetime.datetime.now()
    
    train = pd.read_csv(trainfile, index_col = 0)    
    test = pd.read_csv(testfile, index_col = 0)
    
    y_train = np.array(train['Length of Stay'])

    train = train.drop(columns = ['Length of Stay'])
    
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
            if((t2-t_init).seconds>=30):
                np.savetxt(weightfile,W.reshape(-1,1))
                np.savetxt(outputfile,predictions(X_test,W))
                t_init = datetime.datetime.now()
        #pred = softmax(np.dot(X_train,W),axis=1)
        #while(loss()):
        #    n0 = beta*n0
        #W = W - n0/np.sqrt(i+1)/n*np.dot(X_train.T,np.subtract(pred,y_train_1_hot))
        
    np.savetxt(weightfile,W.reshape(-1,1))
    
    np.savetxt(outputfile,predictions(X_test,W))
    
    
if(sys.argv[1]=="a"):
    part_a(sys.argv[2],sys.argv[3],sys.argv[5],sys.argv[6],sys.argv[4])
elif(sys.argv[1]=="b"):
    part_b(sys.argv[2],sys.argv[3],sys.argv[5],sys.argv[6],sys.argv[4])
elif(sys.argv[1]=="c"):
    part_c(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
else:
    part_d(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])