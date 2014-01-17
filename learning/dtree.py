from learning import Learner
import numpy as np
import scipy.stats
import scoring

class BinaryDecNode:
    """Abstract-type class outline."""
    rule = None
    pos_child = None
    neg_child = None
    isleaf = None
    pass


## DEC TREE DEFINITION ================================================
class HeightLimitedBinaryDecTree(Learner):
    """ Implements a decision tree where training is stopped
    at fixed height. Attributes must be nominal with integer representation.
    Decision rules are simple binary splits on categorical membership."""
    
    def __init__(self, attributes, clabels, impurityfun, hlim):
        Learner.__init__(self, attributes, clabels)
        self.impurityfun = impurityfun
        self.hlim = hlim
        self.root = BinaryDecNode()
        self.unique_attrs = []
        for j in range(self.nattr):
            self.unique_attrs.append(np.unique(self.attributes[:][j]))
        
        self.train()    
        
        
    def train(self):
        objs = range(self.nobjs)
        self.build_dec_tree(self.root,0,objs)
    
    
    def classify(self,attr):
        def classify_inst(obj):
            curnode = self.root
            while not curnode.isleaf:
                if obj[curnode.rule[0]] == curnode.rule[1]:
                    curnode = curnode.pos_child
                else:
                    curnode = curnode.neg_child
            
            return curnode.rule
            
        
        nobj = attr.shape[0]
        pred_class = np.ones(nobj)
        for i in range(nobj):
            pred_class[i] = classify_inst(attr[i])
            
        return pred_class
        

    def build_dec_tree(self, curnode, height, objs):
        if height == self.hlim:
            curnode.isleaf = True
            curnode.rule = int(scipy.stats.mode(self.clabels[objs])[0])
            
        else:
            (attr_ind, split_val, pos_obj) = self.split_node(objs)
            curnode.rule = (attr_ind, split_val)
            curnode.pos_child = BinaryDecNode()
            curnode.neg_child = BinaryDecNode()
            if len(pos_obj) == len(objs):
                # perfect classifier, cannot split further
                curnode.pos_child.is_leaf=True
                curnode.neg_child.is_leaf=False
                pos_class = self.attributes[pos_obj,0]
                neg_class = [x for x in self.unique_classes if x != pos_class][0]
                curnode.pos_child.rule = pos_class
                curnode.neg_child.rule = neg_class
                
            else:
                self.build_dec_tree(curnode.pos_child, height+1, pos_obj)
                pos_obj_set = set(pos_obj)
                self.build_dec_tree(curnode.neg_child, height+1, [x for x in objs if x not in pos_obj_set])
            
    
    def split_node(self, objs):
        attrind = 0
        optimal_split = (None, None, float("-inf"), None)
        for attr in self.unique_attrs:
            for split_val in attr:
                pos_objs = np.argwhere(self.attributes[objs,attrind] == split_val)
                pos_objs = [x for sublist in pos_objs for x in sublist] # reduce pos_objs to single dimension
                objs_set = set(objs)
                
                if len(pos_objs)==0:
                    continue
                
                if len(pos_objs) == len(objs):
                    continue
                
                pos_objs = [x for x in pos_objs if x in objs_set];
                gain = self.calc_gain(objs, pos_objs)
                
                if (attrind, split_val) == (4,6):
                    x = 1
                
                if gain > optimal_split[2]:
                    optimal_split = (attrind, split_val, gain, pos_objs)
                
                if len(attr) == 1:
                    print "Big Problems"
                    
                if len(attr) == 2:
                        break
                
            attrind += 1
            
        return (optimal_split[0], optimal_split[1], optimal_split[3])
    
    
    
    def calc_gain(self,all_objs,pos_objs):
        n_all_objs = len(all_objs)
        n_pos_objs = len(pos_objs)
        n_neg_objs = n_all_objs - n_pos_objs
        all_probs = self.get_cond_probs(all_objs)
        pos_probs = self.get_cond_probs(pos_objs)
        set_pos_objs = set(pos_objs)
        neg_probs = self.get_cond_probs([x for x in all_objs if x not in set_pos_objs])
        par_impurity = self.impurityfun(all_probs)
        pos_impurity = self.impurityfun(pos_probs)
        neg_impurity = self.impurityfun(neg_probs)
        gain = par_impurity-(n_pos_objs*pos_impurity)/n_all_objs-(n_neg_objs*neg_impurity)/n_all_objs
        return gain
    
    
                
    def get_cond_probs(self, objs):
        nobjs = len(objs)
        cprobs = []
        for clab in self.unique_classes:
            cprobs.append(np.sum(self.clabels[objs]==clab)/float(nobjs))
            
        return cprobs
    
    
    ## END DEC TREE DEFINITION ============================================
    
class DecStumpWithInfGain(HeightLimitedBinaryDecTree):
    def __init__(self, attributes, clabels):
        HeightLimitedBinaryDecTree.__init__(self, attributes, clabels, \
                                            impurityfun=scoring.inf_entropy,\
                                            hlim = 1)
        
 
            
    