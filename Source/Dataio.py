import numpy as np
from numpy.matlib import repmat
import sys
import time
from scipy.io import loadmat
from CART import *
from RandomForest import *
from Boosting import *

if __name__ == '__main__':

    # load in some binary test data (labels are -1, +1)
    print('Loading Data...\n')
    data = loadmat("../Dataset/ion.mat")
    xTr  = data['xTr'].T
    yTr  = data['yTr'].flatten()
    xTe  = data['xTe'].T
    yTe  = data['yTe'].flatten()
    
    xTr.shape, yTr.shape, xTe.shape, yTe.shape
    
    ##############################################################################
    # Evaluate the CART trees
    print('\n')
    print('Evaluating CART.. \n')
    t0 = time.time()
    root = cart(xTr, yTr)
    t1 = time.time()
    
    tr_err   = np.mean((predicttree(root,xTr) - yTr)**2)
    te_err   = np.mean((predicttree(root,xTe) - yTe)**2)
    print(predicttree(root,xTe))
    print("elapsed time: %.2f seconds" % (t1-t0))
    print("Training RMSE : %.2f" % tr_err)
    print("Testing  RMSE : %.2f" % te_err)
    ##############################################################################
    # Evaluate the Random Forest
    print('\n')
    print('Evaluating RandomForest.. \n')
    M=20 # max number of trees
    err_trB=[]
    err_teB=[]
    for i in range(M):
        trees=forest(xTr,yTr,i+1)
        trErr = np.mean(np.sign(predictforest(trees,xTr)) != yTr)
        teErr = np.mean(np.sign(predictforest(trees,xTe)) != yTe)
        err_trB.append(trErr)
        err_teB.append(teErr)
        print("[%d]training err = %.4f\ttesting err = %.4f" % (i,trErr, teErr))
    
    ##############################################################################
    # Evaluate AdaBoost
    print('\n')
    print('Evaluating Adaboost.. \n')
    M=20 # max number of trees
    err_trB=[]
    err_teB=[]
    for i in range(M):
        forest,alphas=AdaBoost(xTr,yTr,maxdepth=3,maxiter=i+1)
        trErr = np.mean(np.sign(predictforest(forest,xTr,alphas)) != yTr)
        teErr = np.mean(np.sign(predictforest(forest,xTe,alphas)) != yTe)
        err_trB.append(trErr)
        err_teB.append(teErr)
        print("[%d]training err = %.4f\ttesting err = %.4f" % (i,trErr, teErr))
