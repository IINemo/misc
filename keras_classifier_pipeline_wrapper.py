from keras.wrappers.scikit_learn import KerasClassifier


class KerasClassifierPipelineWrapper(KerasClassifier): 
    ''' This wrapper is indended for sklearn Pipeline. Keras model needs input_shape, 
        however it is not always available prematurely due to feature selection in Pipeline.
        This wrapper determines the input shape and passes it to model construction function.
        Besides it can also deal with sparse matrixes that are feeded to the Pipeline.
    '''   

    def __init__(self, wrapper_shuffle = False, *args, **kwargs):
        super(KerasClassifierPipelineWrapper, self).__init__(*args, **kwargs)
        self.wrapper_shuffle_ = wrapper_shuffle
        self.build_fn_real = None
    
    def fit(self, X, y, **fit_args):
        input_shape = X.shape[1:]
        print input_shape
        
        if self.build_fn_real is None:
            self.build_fn_real = self.build_fn
        
        self.build_fn = lambda : self.build_fn_real(input_shape)
        
        X_fit = X.todense() if scipy.sparse.issparse(X) else X
        if self.wrapper_shuffle_:
            X_fit, y = sklearn.utils.shuffle(X_fit, y)
        
        super(KerasClassifierPipelineWrapper, self).fit(X_fit, y, **fit_args)
        
        return self
    
    def predict(self, X):
        X_pred = X.todense() if scipy.sparse.issparse(X) else X
        return super(KerasClassifierPipelineWrapper, self).predict(X_pred)
    
    def predict_proba(self, X): 
        X_pred = X.todense() if scipy.sparse.issparse(X) else X
        return super(KerasClassifierPipelineWrapper, self).predict_proba(X_pred)

