import csv
import validation.crossval as cv
from learning.boosting import AdaBooster
from learning.dtree import HeightLimitedBinaryDecTree
import learning.scoring as score
import numpy as np


def read_mush_csv(fname):
    f = open(fname, 'rb')
    csvreader = csv.reader(f, delimiter=";")
    mushdat = [];
    clabels = [];
    for row in csvreader:
        clabels.append(row[0])
        row = row[1:11] + row[12:]
        mushdat.append(row)
    
    mushdat = np.array(mushdat, dtype=np.int64)
    clabels = np.array(clabels, dtype=np.int64)
    return (mushdat, clabels)
    

class DecStumpIG(HeightLimitedBinaryDecTree):
    def __init__(self, attributes, clabels):
        HeightLimitedBinaryDecTree.__init__(self, attributes, clabels, impurityfun=score.inf_entropy, hlim=1)


class DTTwoLayGini(HeightLimitedBinaryDecTree):
    def __init__(self, attributes, clabels):
        HeightLimitedBinaryDecTree.__init__(self, attributes, clabels, impurityfun=score.gini, hlim=2)


class AdaBoost(AdaBooster):
    def __init__(self, attributes, clabels, T):
        AdaBooster.__init__(self, attributes, clabels, DecStumpIG, T)


def myAdaBoost(filename, T):
    class AB(AdaBoost):
        def __init__(self, attributes, clabels):
            AdaBoost.__init__(self, attributes, clabels, T)
            
    (attributes, clabels) = read_mush_csv(filename)
    print "ADABOOST WITH " + str(T) + " STUMPS ON FILE: " + filename + " --------------"
    cv.perform_k_fold_cv(AB, attributes, clabels, k=10, verbose=True)


def dstumpIG(filename):
    (attributes, clabels) = read_mush_csv(filename)
    print "DECISION STUMP ON FILE: " + filename + " --------------"
    cv.perform_k_fold_cv(DecStumpIG, attributes, clabels, k=10, verbose=True)
    
    
def dtree2IG(filename):
    (attributes, clabels) = read_mush_csv(filename)
    print "TWO LAYER TREE ON FILE: " + filename + " --------------"
    cv.perform_k_fold_cv(DTTwoLayGini, attributes, clabels, k=10, verbose=True)
    
