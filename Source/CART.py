import numpy as np

class TreeNode(object):
    """Tree class.
    
    (You don't need to add any methods or fields here but feel
    free to if you like. Our tests will only reference the fields
    defined in the constructor below, so be sure to set these
    correctly.)
    """
    
    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction
        

def sqsplit(xTr,yTr,weights=[]):
    """Finds the best feature, cut value, and loss value.
    
    Input:
        xTr:     n x d matrix of data points
        yTr:     n-dimensional vector of labels
        weights: n-dimensional weight vector for data points
    
    Output:
        feature:  index of the best cut's feature
        cut:      cut-value of the best cut
        bestloss: loss of the best cut
    """
    N,D = xTr.shape
    assert D > 0 # must have at least one dimension
    assert N > 1 # must have at least two samples
    if weights == []: # if no weights are passed on, assign uniform weights
        weights = np.ones(N)
    weights = weights/sum(weights) # Weights need to sum to one (we just normalize them)
    
    # construct the loss matrix
    
    loss = np.zeros((N-1,D))
    
    # calculate Q
    Q = np.dot(weights,yTr**2)
    # initialize bestloss, cut
    feature = 0
    bestloss = Q
    cut = 0
    # for each split calculate W_L and W_R
    for d in range(D):
        x = xTr[:,d].flatten()
        idx = np.argsort(x)
        x = x[idx]
        w = weights[idx]
        y = yTr[idx]
        #initialize W and R
        W_L = 0.
        W_R = 1.
        P_L = 0.
        P_R = np.dot(weights,yTr)
       
        
        #update W R
        for k in range(N-1):
            W_L = W_L + w[k]
            W_R = W_R - w[k]
            P_L = P_L + w[k]*y[k]
            P_R = P_R - w[k]*y[k]
            if x[k]==x[k+1]: 
                continue
            else: 
                loss = Q - P_L**2/W_L - P_R**2/W_R
                if loss < bestloss:
                    bestloss = loss
                    feature = d
                    cut = x[k]
                
    
    return feature, cut, bestloss

def cart(xTr,yTr,depth=np.inf,weights=None):
    """Builds a CART tree.
    
    The maximum tree depth is defined by "maxdepth" (maxdepth=2 means one split).
    Each example can be weighted with "weights".

    Args:
        xTr:      n x d matrix of data
        yTr:      n-dimensional vector
        maxdepth: maximum tree depth
        weights:  n-dimensional weight vector for data points

    Returns:
        tree: root of decision tree
    """
    n,d = xTr.shape
    if weights is None:
        w = np.ones(n) / float(n)
        #w = np.ones(n) / n
    else:
        w = weights
    if depth == np.inf: depth = n-1
    #tree = None
    
    #tree = treeNode(left, right, parent, cutoff_id, cutoff_val, prediction)
    

    #prediction
    prediction = np.mean(yTr)
    x_u= np.vstack(set(map(tuple, xTr))).shape[0]
    y_u = len(np.unique(yTr))
    if depth == 1 or x_u==1 or y_u == 1:
        tree = TreeNode(None, None, None, None,None, prediction)
    else:
        feature, cut, bestloss = sqsplit(xTr, yTr, w)
        
        # generate left and right branch
        xTr_l = xTr[xTr[:,feature]<=cut,:]
        xTr_r = xTr[xTr[:,feature]>cut,:]
        yTr_l = yTr[xTr[:,feature]<=cut]
        yTr_r = yTr[xTr[:,feature]>cut]
        w_l = w[xTr[:,feature]<=cut]
        w_l = w_l/sum(w_l)
        w_r = w[xTr[:,feature]>cut]
        w_r = w_r/sum(w_r)
        
        tree = TreeNode(None, None, None, feature, cut, prediction)
        tree.left = cart(xTr_l,yTr_l,depth-1,w_l)
        tree.right = cart(xTr_r,yTr_r,depth-1,w_r)
        tree.right.parent = tree
        tree.left.parent = tree
    
    return tree

def predicttree(root,xTe):
    """Evaluates xTe using decision tree root.
    
    Input:
        root: TreeNode decision tree
        xTe:  n x d matrix of data points
    
    Output:
        pred: n-dimensional vector of predictions
    """
    assert root is not None
    n,d = xTe.shape
    pred = np.zeros(n)
    for i in range(n):
        r = root
        x = xTe[i,:].flatten()
        while r.left != None:
            feature = r.cutoff_id
            cut = r.cutoff_val
            if x[feature]<=cut:
                r = r.left
            else:
                r = r.right
        else:
            pred[i] = r.prediction
    
    return pred

