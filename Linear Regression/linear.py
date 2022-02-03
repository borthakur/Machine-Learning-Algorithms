import pandas as pd
import numpy as np
import sys


def part_a(trainfile, testfile, outputfile, weightfile):
    df = pd.read_csv(trainfile,index_col=[0])
    
    X = df.to_numpy()
    y = X[:,-1]
    X = X[:,:-1]
    n,m = X.shape
    y.shape = (n,1)

    ones = np.ones((n,1))
    X = np.append(ones,X,axis=1)

    w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    w.shape = (w.shape[0],1)
    np.savetxt(weightfile,w)

    df2 = pd.read_csv(testfile,index_col=[0])

    X_test = df2.to_numpy()
    n_test,m = X_test.shape

    ones = np.ones((n_test,1))
    X_test = np.append(ones,X_test,axis=1)

    y_pred = np.dot(X_test,w)
    np.savetxt(outputfile,y_pred)

def part_b(trainfile, testfile, regularization, outputfile, weightfile, bestparameter):
    
    df = pd.read_csv(trainfile,index_col=[0])

    X = df.to_numpy()
    y = X[:,-1]
    X = X[:,:-1]
    n,m = X.shape
    y.shape = (n,1)

    ones = np.ones((n,1))
    X = np.append(ones,X,axis=1)

    def loss(y_pred,y_test):
        diff = np.subtract(y_pred,y_test)
        return np.sqrt(np.dot(diff.T,diff)/np.dot(y_test.T,y_test))

    def performance(X,y,X_test,y_test,lam_da):
        w = np.dot(np.linalg.inv(np.add(np.dot(X.T,X),lam_da*np.identity(m+1))),np.dot(X.T,y))
        w.shape = (w.shape[0],1)
        y_pred = np.dot(X_test,w)
        return loss(y_pred,y_test)

    lam_da_file = open(regularization, "r")
    lams = lam_da_file.read()
    lam_das = lams.split(",")
    lam_da_file.close()
    lam_das = [float(lam_da) for lam_da in lam_das]
    
    def do_k_fold(X,y,lam_das,k_fold):
        n = X.shape[0]
        k_fold_indices = []
        diff = n//k_fold
        for i in range(0,k_fold-1):
            k_fold_indices += [range(i*diff,(i+1)*diff)]
        k_fold_indices += [range((k_fold-1)*diff,n)]
        X_splits = []
        y_splits = []
        for i in range(0,k_fold):
            X_splits += [X[k_fold_indices[i]]]
            y_splits += [y[k_fold_indices[i]]]
    
        for lam_da in lam_das[0:1]:
            X_test_fold = X_splits[0]
            y_test_fold = y_splits[0]
    
            X_fold = np.concatenate(X_splits[1:])
            y_fold = np.concatenate(y_splits[1:])
    
            rms_sum = performance(X_fold,y_fold,X_test_fold,y_test_fold,lam_da)
        
            for i in range(1,k_fold):
                X_test_fold = X_splits[i]
                y_test_fold = y_splits[i]
                if(i!=k_fold-1):
                    X_fold_a = np.concatenate(X_splits[:i])
                    X_fold_b = np.concatenate(X_splits[i+1:])
                    X_fold = np.concatenate([X_fold_a,X_fold_b])
                    y_fold_a = np.concatenate(y_splits[:i])
                    y_fold_b = np.concatenate(y_splits[i+1:])
                    y_fold = np.concatenate([y_fold_a,y_fold_b])
                else:
                    X_fold = np.concatenate(X_splits[:i])
                    y_fold = np.concatenate(y_splits[:i])
                rms_sum += performance(X_fold,y_fold,X_test_fold,y_test_fold,lam_da)
            best_avg_rms = rms_sum/k_fold
            best_lam_da = lam_da
            
        for lam_da in lam_das[1:]:
            X_test_fold = X_splits[0]
            y_test_fold = y_splits[0]
        
            X_fold = np.concatenate(X_splits[1:])
            y_fold = np.concatenate(y_splits[1:])
        
            rms_sum = performance(X_fold,y_fold,X_test_fold,y_test_fold,lam_da)
        
            for i in range(1,k_fold):
                X_test_fold = X_splits[i]
                y_test_fold = y_splits[i]
                if(i!=k_fold-1):
                    X_fold_a = np.concatenate(X_splits[:i])
                    X_fold_b = np.concatenate(X_splits[i+1:])
                    X_fold = np.concatenate([X_fold_a,X_fold_b])
                    y_fold_a = np.concatenate(y_splits[:i])
                    y_fold_b = np.concatenate(y_splits[i+1:])
                    y_fold = np.concatenate([y_fold_a,y_fold_b])
                else:
                    X_fold = np.concatenate(X_splits[:i])
                    y_fold = np.concatenate(y_splits[:i])
                rms_sum += performance(X_fold,y_fold,X_test_fold,y_test_fold,lam_da)
            avg_rms = rms_sum/k_fold
            
            if(avg_rms<best_avg_rms):
                best_lam_da = lam_da
                best_avg_rms = avg_rms
        return best_lam_da
    
    k_fold = 10

    best_lam_da = do_k_fold(X,y,lam_das,k_fold)
    
    w = np.dot(np.linalg.inv(np.add(np.dot(X.T,X),best_lam_da*np.identity(m+1))),np.dot(X.T,y))
    np.savetxt(weightfile,w)
    
    df2 = pd.read_csv(testfile,index_col=[0])

    X_test = df2.to_numpy()
    n_test,m = X_test.shape
    
    ones = np.ones((n_test,1))
    X_test = np.append(ones,X_test,axis=1)
    
    y_pred = np.dot(X_test,w)
    np.savetxt(outputfile,y_pred)
    
    lam_da_file = open(bestparameter, "w")
    lam_da_file.write(str(best_lam_da))
    lam_da_file.close()
    
    
def part_c(trainfile, testfile, outputfile):
    
    df_train = pd.read_csv(trainfile,index_col=[0])
    df_test = pd.read_csv(testfile,index_col=[0])
    
    y = df_train["Total Costs"].to_numpy()
    del df_train["Total Costs"]
    
    def create_feature(lens, categ, element):
        categ[categ!=element] = 0
        categ[categ!=0] = 1
        return lens*categ
    
    lens_train = df_train["Length of Stay"].to_numpy()
    lens_test = df_test["Length of Stay"].to_numpy()
    
    df_train["Ones"] = np.ones(len(lens_train))
    df_test["Ones"] = np.ones(len(lens_test))
    
    def add_feature(col_name,element):
        train_col = np.copy(df_train[col_name].to_numpy())
        test_col = np.copy(df_test[col_name].to_numpy())
        
        df_train[col_name+str(element)] = create_feature(lens_train,train_col,element)
        df_test[col_name+str(element)] = create_feature(lens_test,test_col,element)
        
    def del_feature(col_name):
        del df_train[col_name]
        del df_test[col_name]
        
    def feature_change(col_name, elements):
        for element in elements:
            add_feature(col_name,element)
        del_feature(col_name)
        
        
    curr_col_name = "Health Service Area"
    elements = [4,5,6,7,8]
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Hospital County"
    elements = [2,3,4,8, 13, 19,23, 26, 29, 30, 33, 36, 38, 41, 46,48, 54, 55]
    feature_change(curr_col_name,elements)
        
    curr_col_name = "Operating Certificate Number"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Facility Id"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Facility Name"
    elements = [1, 2, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 24, 26,27, 28, 35, 40, 41, 45,  48, 49, 52, 54, 57, 58,59,60,61,62,63,64,66,67,68,70,71,72,74,76,77,78,79,84,85,86,87,89,90,91,94,95,96,97,99,100,102,103,104,106,107,108,109,111,112,114,115,116,117,118,120,121,122,123,126,127,130,133,134,135,136,139,140,141,143,145,147,148,149,150,151,152,153,154,156,161,162,164,166,167,168,169,170,174,175,176,177,181,182,184,185,186,189,190,191,193,195,197,198,199,200,201,202,205,206,207,208,209,210,211]
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Age Group"
    elements = [1,2,3,5]
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Zip Code - 3 digits"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Gender"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Race"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Ethnicity"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Type of Admission"
    elements = [1,3,5,6]
    feature_change(curr_col_name,elements)
    
    curr_col_name ="Patient Disposition"
    elements = [2,3,5,7,8,11,13,14,16,17,18,19]
    feature_change(curr_col_name,elements)
    
    curr_col_name = "CCS Diagnosis Code"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "CCS Diagnosis Description"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "CCS Procedure Code"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "CCS Procedure Description"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "APR DRG Code"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "APR DRG Description"
    elements = []
    feature_change(curr_col_name,elements)    
    
    
    curr_col_name = "APR MDC Code"
    elements = []
    feature_change(curr_col_name,elements)
  
    
    curr_col_name = "APR MDC Description"
    elements = [1,2,3,4,6,7,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24]
    feature_change(curr_col_name,elements)    
    
    
    curr_col_name = "APR Severity of Illness Code"
    elements = []
    feature_change(curr_col_name,elements)

    
    curr_col_name = "APR Severity of Illness Description"
    elements = [1,2,3]
    feature_change(curr_col_name,elements)
    
        
    curr_col_name = "APR Risk of Mortality"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "APR Medical Surgical Description"
    elements = [2]
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Payment Typology 1"
    elements = [1,6,7,8,9,10]
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Payment Typology 2"
    elements = [1,3,5,6,9,10,11]
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Payment Typology 3"
    elements = [4,6,8,11]
    feature_change(curr_col_name,elements)
    
    curr_col_name ="Birth Weight"
    elements = []
    feature_change(curr_col_name,elements)
    
    curr_col_name = "Emergency Department Indicator"
    elements = [1]
    feature_change(curr_col_name,elements)
    
    X = df_train.to_numpy()
    n,m = X.shape
    y = y.reshape(n,1)

    X_test = df_test.to_numpy()
    n_test,m = X_test.shape
    
    
    
    lam_da = 0.001
    w = np.dot(np.linalg.inv(np.add(np.dot(X.T,X),lam_da*np.identity(X.shape[1]))),np.dot(X.T,y))
    
    y_pred = np.dot(X_test,w)
    np.savetxt(outputfile,y_pred)
    
if(sys.argv[1]=="a"):
    part_a(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
elif(sys.argv[1]=="b"):
    part_b(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7])
else:
    part_c(sys.argv[2],sys.argv[3],sys.argv[4])