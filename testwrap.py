from hw2 import read_mush_csv as rmc
import learning.scoring as score
from learning.dtree import HeightLimitedBinaryDecTree, DecStumpWithInfGain
from learning.boosting import AdaBooster
from numpy.random import randint as rint
import numpy as np
import validation.crossval as cv
from hw2 import myAdaBoost

(mushdat, clabels) = rmc("Mushroom.csv")

myAdaBoost("Mushroom.csv", 20)

# for x in range(10):
#     pass
#     #model = AdaBooster(mushdat, clabels, DecStumpWithInfGain, x)
#     #model = HeightLimitedBinaryDecTree(mushdat, clabels, hlim = 1, impurityfun = score.rand)
# 
# class AdaBoost(AdaBooster):
#     def __init__(self, attributes, clabels):
#         AdaBooster.__init__(self, attributes, clabels, DecStumpWithInfGain, T=20)
# 
# class TwoLay(HeightLimitedBinaryDecTree):
#     def __init__(self, attributes, clabels):
#         HeightLimitedBinaryDecTree.__init__(self, attributes, clabels, hlim=2, impurityfun = score.gini)
#         
# cv.perform_k_fold_cv(AdaBoost, mushdat, clabels, verbose=True)
# results[0].printstats()
# print results

#model = DecStumpWithInfGain(mushdat, clabels)
# model = TwoLay(mushdat, clabels)
# x=2

# cvinds = cv.CVPartInds(len(mushdat))
# for x in range(10):
#     (testinds, traininds) = cvinds.next()
#     trainattr = mushdat[traininds]
#     trainclabs = clabels[traininds]
#     testattr = mushdat[testinds]
#     testclabs = clabels[testinds]
#     #model = AdaBooster(trainattr, trainclabs, DecStumpWithInfGain, T=50)
#     model = DecStumpWithInfGain(trainattr, trainclabs)
#     trainpreds = model.classify(trainattr)
#     trainerr = np.sum(np.array(trainpreds!=trainclabs, dtype=np.int))/float(len(trainpreds))
#     testpreds = model.classify(testattr)
#     testerr = np.sum(np.array(testpreds!=testclabs, dtype=np.int))/float(len(testpreds))
#     
#     print "--"
#     print trainerr
#     print testerr



# for stumpcount in [5,10,20,40]:
#     class AdaBoost(AdaBooster):
#         def __init__(self, attributes, clabels):
#             AdaBooster.__init__(self, attributes, clabels, DecStumpWithInfGain, T=stumpcount)
# 
#     print "WITH " + str(stumpcount) + " STUMPS -----------"
#     stats = cv.perform_k_fold_cv(AdaBoost, mushdat, clabels)
#     stats.printstats()
