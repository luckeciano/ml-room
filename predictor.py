
class Predictor():

    def __init__(self):
        self.predictor = None
    
    def train(self,train_x, train_y):
        raise NotImplementedError()

    def predict(self,test_x):
        raise NotImplementedError()
    