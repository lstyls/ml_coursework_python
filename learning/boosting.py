from learning import Learner
import numpy as np

class AdaBooster(Learner):
    
    def __init__(self, attributes, clabels, base_weak_learner, T, printupdates=False):
        Learner.__init__(self, attributes, clabels)
        self.alphas = np.zeros(T, dtype=np.float)
        self.bwl = base_weak_learner
        self.T = T
        self.classifiers = []
        self.pud = printupdates
        self.sampsize = self.nobjs/5
        self.train()
    
    
    def train(self):
        t = 0
        weights = self.init_weights()
        while t<self.T:
            sampidx = np.random.choice(np.arange(self.nobjs), p=weights, size=self.sampsize)
            attrs = self.attributes[sampidx]
            clabs = self.clabels[sampidx]
            wl = self.bwl(attrs,clabs)
            pred_classes = wl.classify(self.attributes)
            misses = (pred_classes != self.clabels)
            err = self.calc_error(misses, weights)
            if err > 0.5:
                weights = self.init_weights()
                continue
            
            self.classifiers.append(wl)
            self.alphas[t] = 0.5*np.log((1-err)/err)
            weights = self.update_weights(misses, self.alphas[t], weights)
            t += 1
    
    def test_classify(self, testattr, testclabels):
        n_testobjs = testattr.shape[0]
        pred_classes = np.ones((self.T, n_testobjs))
        i = 0
        for wl in self.classifiers:
            if self.pud:
                print "Classifying on wl " + str(i+1)
            pred_classes[i] = wl.classify(testattr)
            i += 1
        
        errates = []
        for predictor in pred_classes:
            matches = predictor == testclabels
            sum_matches = np.sum(np.array(matches,dtype=np.int))
            errate = 1 - sum_matches/float(len(matches))
            errates.append(errate)
        
        pred_pos = np.array(pred_classes == self.unique_classes[0], dtype=np.int)
        pred_neg = -np.array(pred_classes == self.unique_classes[1], dtype=np.int)
        pred_classes = pred_pos + pred_neg;
        if not np.all(np.logical_or(pred_classes == -1, pred_classes == 1)):
            print "big problems"
            
        pred_classes = np.dot(self.alphas, pred_classes)
        pred_classes = np.array(pred_classes > 0, dtype=np.int)
        pred_classes = self.unique_classes[pred_classes]
        return pred_classes
    
    def classify(self, testattr):
        n_testobjs = testattr.shape[0]
        pred_classes = np.ones((self.T, n_testobjs))
        i = 0
        for wl in self.classifiers:
            pred_classes[i] = wl.classify(testattr)
            i += 1
        
        pred_pos = np.array(pred_classes == self.unique_classes[0], dtype=np.int)
        pred_neg = -np.array(pred_classes == self.unique_classes[1], dtype=np.int)
        pred_classes = pred_pos + pred_neg;
            
        pred_classes = np.dot(self.alphas, pred_classes)
        pred_classes = np.array(pred_classes < 0, dtype=np.int)
        pred_classes = self.unique_classes[pred_classes]
        return pred_classes
            
    
    def calc_error(self, misses, weights):
        err = np.sum(weights[misses])
        return err
    
    
    def init_weights(self):
        return np.ones(self.nobjs, dtype=np.float64)/float(self.nobjs)
    
    
    def update_weights(self, misses, alpha, weights):
        exp_vals = alpha*(np.array(misses, dtype=np.int)-np.array(~misses, dtype=np.int))
        update = np.exp(exp_vals)
        weights = np.multiply(weights,update)
        weights = weights/np.sum(weights)
        return weights
    
        