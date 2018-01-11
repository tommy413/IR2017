class IRmodel(object):
    def __init__(self):
        model = None

    def get_model():
        raise NotImplementedError("Subclasses should implement this!")

    def load_model():
        raise NotImplementedError("Subclasses should implement this!")

    def fit(self,X,Y):
        raise NotImplementedError("Subclasses should implement this!")

    def predict(self,X):
        raise NotImplementedError("Subclasses should implement this!")
