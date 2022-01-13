from . import Processor


class SGD:
    def __init__(self, learning_rate=0.0015):
        self.learning_rate = learning_rate

    def update(self, x, gradient):
        # new position from given gradient
        x_new = Processor.__update__(x, self.learning_rate, gradient)
        return x_new