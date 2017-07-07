import numpy as np
from CART import *

def AdaBoost(x,y,maxiter=100,maxdepth=2):
    """Learns a boosted decision tree.
    
    Input:
        x:        n x d matrix of data points
        y:        n-dimensional vector of labels
        maxiter:  maximum number of trees
        maxdepth: maximum depth of a tree
        
    Output:
        forest: list of TreeNode decision trees of length m
        alphas: m-dimensional weight vector
        
    (note, m is at most maxiter, but may be smaller,
    as dictated by the Adaboost algorithm)
    """
    assert np.allclose(np.unique(y), np.array([-1,1])); # the labels must be -1 and 1 
    n,d = x.shape
    weights = np.ones(n) / n
    alphas = list()
    forest = list()
    for t in range(maxiter):
        tree = cart(x,y,maxdepth,weights)
        pr = np.sign(predicttree(tree,x))
        eps = np.sum(weights[pr!=y])
        print("eps%s:"%eps)
        if eps < 0.5:       
            thing = (1-eps)/eps
            thing2 = (1-eps)*eps
            alpha = 0.5*np.log(thing)
            alphas.append(alpha)
            forest.append(tree)
            weights = weights*np.exp(-alpha*y*pr)/(2*np.sqrt(thing2))
        else:
            break
                          
    return forest, np.array(alphas)
