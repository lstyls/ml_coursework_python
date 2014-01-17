import numpy as np

def inf_entropy(cond_probs):
    entropy = 0
    for p in cond_probs:
        if p != 0:
            entropy -= p*np.log2(p)
            
    return entropy
            

def gini(cond_probs):
    return 1-np.sum(pow(np.array(cond_probs),2))

def rand(cond_probs):
    return np.random.rand()