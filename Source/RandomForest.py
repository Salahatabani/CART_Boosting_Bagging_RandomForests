import numpy as np
from CART import *
inf = np.inf
def forest(xTr, yTr, m, maxdepth=inf):
    """Creates a random forest.
    
    Input:
        xTr:      n x d matrix of data points
        yTr:      n-dimensional vector of labels
        m:        number of trees in the forest
        maxdepth: maximum depth of tree
        
    Output:
        trees: list of TreeNode decision trees of length m
    """
    trees = list()
    n, d = xTr.shape
    for i in range(m):
        rand = np.random.choice(n,n)
        new_xTr = xTr[rand,:]
        new_yTr = yTr[rand]
        trees.append(cart(new_xTr,new_yTr,depth=maxdepth))
    return trees

def predictforest(trees, X, alphas=None):
    """Evaluates X using trees. Can be used for random forest trees or boosted trees
    
    Input:
        trees:  list of TreeNode decision trees of length m
        X:      n x d matrix of data points
        alphas: m-dimensional weight vector
        
    Output:
        pred: n-dimensional vector of predictions
    """
    m = len(trees)
    n,d = X.shape
    if alphas is None:
        alphas = np.ones(m) / len(trees)
            
    p = np.zeros((m,n))
    
    for t in range(m):
        p[t,:] = predicttree(trees[t],X)
    
    pred = np.inner(p.T,alphas).flatten()
    return pred        