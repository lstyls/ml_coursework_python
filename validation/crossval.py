from numpy.random import choice
import numpy as np
from validation import perform_holdout_validation as holdout
#from validation import perform_test_holdout_validation as test_holdout

class CVPartInds():
    """Store indices of random partitions for k-fold cross validation.
    Basic functionality equivalent to the MATLAB crossvalind."""
    def __init__(self, n, k=10):
        self.n = n
        self.partinds = cvind(n,k)
    
    def __iter__(self):
        return self
    
    def next(self):
        if not self.partinds:
            raise StopIteration
        
        testind = self.partinds.pop()
        trainind = [x for x in range(self.n) if x not in set(testind)]
        return(testind, trainind)
    
    
class CVStats():
    """Calculate and store train and test statistics for k-fold crossvalidation."""
    def __init__(self, trainerr, testerr):
        self.avg_testerr = np.mean(testerr)
        self.avg_trainerr = np.mean(trainerr)
        self.std_testerr = np.std(testerr)    
        self.std_trainerr = np.std(trainerr)
    
    def printstats(self):
        print "Test Error | Average: " + str(self.avg_testerr) \
                        + ", Std: " + str(self.std_testerr)
        print "Train Error | Average: " + str(self.avg_trainerr) \
                        + ", Std: " + str(self.std_trainerr)
        

def cvind(n,k):
    """Function generates indices for k equally sized partitions of data indices."""
    partitions = []
    inds = range(n)
    while k:
        part = choice(inds, np.ceil(len(inds)/float(k)), replace = False)
        inds = [x for x in inds if x not in set(part)]
        partitions.append(part)
        k -= 1
        
    return partitions


def partition_data(k, *mats):
    """ Partitions along first dimension."""
    partinds = cvind(len(mats[0]), k)
    partitions = [[mat[partind] for partind in partinds] for mat in mats]
    if len(partitions) == 1:
        partitions = partitions[0]
    return partitions


def perform_k_fold_cv(LearningModel, attributes, clabels, k=10, verbose=True):
    """Perform k-fold crossvalidation using a specific learning model on dataset and print results to stdout.
    
    Inputs:
        LearningModel - Training model object of descendant type of the abstract learning.Learner class.
        
        attributes - M-by-N numpy array of numerical data, where M is the number of objects and N the number of attributes.
        
        clabels - M-by-1 numpy array of class labels, such that clabels[i] is the class label for the object referenced in attributes[i,:].
        
        k - Number of partitions for crossvalidation.
        
        verbose - Boolean. If true, print updates and detailed statistics.
    """
    
    
    nattr = len(attributes)
    testerr = []
    trainerr = []
    cvpartitions = CVPartInds(nattr, k)
    fold = 1;
    if verbose:
        print "Performing K-Fold Cross-Validation"
        
    for (testind, trainind) in cvpartitions:
        fold += 1
        trainattr = attributes[trainind]
        trainclabs = clabels[trainind]
        testattr = attributes[testind]
        testclabs = clabels[testind]
        (trerr, tsterr) = holdout(LearningModel, trainattr, testattr, trainclabs, testclabs)
        trainerr.append(trerr)
        testerr.append(tsterr)
        
        print "on fold: " + str(fold-1) + " | Training Error = " + str(trerr) + "; Test Error = " + str(tsterr)
        
    testerr = np.array(testerr)
    trainerr = np.array(trainerr)
    
    if verbose:
        print " "
        print "Stats Summary:"
        stats = CVStats(trainerr, testerr)
        stats.printstats()
        print " "
    
    #return CVStats(trainerr, testerr)
    
# def test_perform_k_fold_cv(LearningModel, attributes, clabels, k=10, verbose=False):
#    """ A testing function."""
#     nattr = len(attributes)
#     testerr = []
#     trainerr = []
#     cvpartitions = CVPartInds(nattr, k)
#     fold = 1;
#     print "Performing K-Fold Cross-Validation"
#     for (testind, trainind) in cvpartitions:
#         print "    Fold = " + str(fold)
#         fold += 1
#         trainattr = attributes[trainind]
#         trainclabs = clabels[trainind]
#         testattr = attributes[testind]
#         testclabs = clabels[testind]
#         (trerr, tsterr) = test_holdout(LearningModel, trainattr, testattr, trainclabs, testclabs)
#         trainerr.append(trerr)
#         testerr.append(tsterr)
#         
#     testerr = np.array(testerr)
#     trainerr = np.array(trainerr)
#     
#     if verbose:
#         for fold in range(k):
#             
#         return (CVStats(trainerr, testerr), trainerr, testerr)
#     
#     else:
#         return CVStats(trainerr, testerr)
    
        
    
    