from numpy import sum

def perform_holdout_validation(LearningModel, trainattr, testattr, trainclabs, testclabs):
    model = LearningModel(trainattr, trainclabs)
    pred_train = model.classify(trainattr)
    trerr = sum(pred_train != trainclabs)/float(len(pred_train))
    pred_test = model.classify(testattr)
    tsterr = sum(pred_test != testclabs)/float(len(pred_test))
    return (trerr, tsterr)

def perform_test_holdout_validation(LearningModel, trainattr, testattr, trainclabs, testclabs):
    model = LearningModel(trainattr, trainclabs)
    pred_train = model.test_classify(trainattr, trainclabs)
    trerr = sum(pred_train == trainclabs)/float(len(pred_train))
    pred_test = model.test_classify(testattr, testclabs)
    tsterr = sum(pred_test == testclabs)/float(len(pred_test))
    return (trerr, tsterr)