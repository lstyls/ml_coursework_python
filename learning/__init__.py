from numpy import unique

class Learner:
    """To be used as an abstract class for classes that implement
    machine learning models"""
    
    def __init__(self, attributes, clabels):
        (self.nobjs, self.nattr) = attributes.shape
        self.attributes = attributes
        self.clabels = clabels
        self.unique_classes = unique(clabels)
        
    def train(self):
        pass
    
    def classify(self,testattr):
        pass