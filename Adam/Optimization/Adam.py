from . import Processor


class Adam:
    def __init__(self, learning_rate=0.5, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.gradient_moment1 = 0
        self.gradient_moment2 = 0
        self.gradient_moment = 0
        self.n = 1

    def update(self, x, gradient):
        self.gradient_moment1 = Processor.__weighted_average__(self.beta1, self.gradient_moment1, gradient)
        self.gradient_moment2 = Processor.__weighted_average__(self.beta2, self.gradient_moment2, gradient ** 2)
        new_learning_rate = Processor.__update_learning_rate__(self.learning_rate, self.beta1**self.n, self.beta2**self.n)
        gradient_reciprocal_sqrt = Processor.__reciprocal_sqrt__(self.gradient_moment2)
        x_new = Processor.__update__(x, gradient_reciprocal_sqrt*new_learning_rate, self.gradient_moment1)

        self.n += 1


        return x_new