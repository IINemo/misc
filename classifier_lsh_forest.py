from sklearn.neighbors import LSHForest
from sklearn.base import ClassifierMixin
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class ClassifierLSHForest(ClassifierMixin, LSHForest):
    """ 
    KNeighborsClassifier with partial_fit method for online learning.
    Memory-based classifier. Wrapper around LSHForest.
    """

    def __init__(self,
                 n_estimators = 10, 
                 radius = 1.0, 
                 n_candidates = 50, 
                 n_neighbors = 5, 
                 min_hash_match = 4, 
                 radius_cutoff_ratio = 0.9, 
                 random_state = None,
                 class_weights = None):
        self.lshf_ = LSHForest(n_estimators = n_estimators,
                             radius = radius,
                             n_candidates = n_candidates, 
                             n_neighbors = n_neighbors,
                             min_hash_match = min_hash_match,
                             radius_cutoff_ratio = radius_cutoff_ratio,
                             random_state = random_state) 
        self.y_ = None
        self.classes_ = list()
        self.class_weights_ = class_weights
    
    def fit(self, X, y):
        self.y_ = y
        self.classes_ = np.unique(y).tolist()
        self.lshf_.fit(X)
        
        print 'fitted'
        return self
    
    def partial_fit(self, X, y, *args, **kwargs):
        if self.y_ is None:
            self.y_ = y
        else:
            self.y_ = np.concatenate((self.y_, y))
            print self.y_.shape
            
        for yi in y:
            if yi not in self.classes_:
                self.classes_.append(yi)
        
        self.lshf_.partial_fit(X)
        
        return self
    
    def _kernel(self, x):
        return np.exp(-x)
    
    def _get_class_weights(self):
        return compute_class_weight(self.class_weights_, self.classes_, self.y_)
    
    def _compute_weights(self, X):
        dists, neighbors = self.lshf_.kneighbors(X, return_distance = True)
        
        result = np.zeros((neighbors.shape[0], len(self.classes_)))
        for i in xrange(neighbors.shape[0]):
            for cl_index, cl in enumerate(self.classes_):
                result[i, cl_index] = self._kernel(dists[i][self.y_[neighbors[i]] == cl]).sum()
                
        if self.class_weights_ is not None:
            result *= self._get_class_weights()
                
        return result
    
    def predict(self, X):
        weights = self._compute_weights(X)
        result = np.argmax(weights, axis = 1)
        for i in xrange(X.shape[0]):
            result = self.classes_[result[i]]
            
        return result
    
    def predict_proba(self, X):
        weights = self._compute_weights(X)
        
        normalizer = weights.sum(axis = 1)
        normalizer[normalizer == 0.0] = 1.0
        
        weights /= normalizer
        
        return weights
