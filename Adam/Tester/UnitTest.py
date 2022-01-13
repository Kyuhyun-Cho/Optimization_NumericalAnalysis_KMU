import tensorflow as tf
import numpy as np
import Optimization.Optimization as Optimization
from Tester.Drawer import *
import sys

class Model(object):
    def __init__(self, a, b, init):
        self.x = tf.Variable(float(init[0]))
        self.y = tf.Variable(float(init[1]))
        self.a = tf.Variable(float(a))
        self.b = tf.Variable(float(b))

    def __call__(self):
        return (self.a - self.x) ** 2 + self.b * (self.y - self.x ** 2) ** 2


def test_Qualitative(optimizer, f, init):
    print(type(optimizer).__name__)
    Opt = Optimization.Optimization(optimizer=optimizer, function=f)
    result, iter_X, iter_Y, iter_cnt = Opt.optimize(init)
    print(f"position (x,y) : {result}, z: {iter_Y[-1]:.3f}, iteration:{iter_cnt}")
    show(f, iter_X, iter_Y, iter_cnt)



def test_Quantitative(my_optimizer, func, init):
    print(type(my_optimizer).__name__ )
    model = Model(func.a, func.b, init)

    if type(my_optimizer).__name__ == "SGD":
        optimizer = tf.optimizers.SGD(learning_rate=my_optimizer.learning_rate)
    elif type(my_optimizer).__name__ == "Adam":
        optimizer = tf.optimizers.Adam(learning_rate=my_optimizer.learning_rate, beta_1=my_optimizer.beta1, beta_2=my_optimizer.beta2, epsilon=sys.float_info.epsilon)
    else:
        return

    Opt = Optimization.Optimization(optimizer=my_optimizer, function=func)
    my_output = Opt.optimize(init)

    def train(model):
        with tf.GradientTape() as t:
            prev_z = model()
        grads = t.gradient(prev_z, [model.x, model.y])
        optimizer.apply_gradients(zip(grads, [model.x, model.y]))
        z = model()
        diff = abs(prev_z.numpy() - z.numpy())
        if diff < Opt.early_stop or np.isnan(diff):
            return [model.x.numpy(), model.y.numpy(), z.numpy()]
        return 0

    for i in range(Opt.iter_max):
        output = train(model)
        if output != 0:
            break

    print(f"position (x,y) : {output[:2]}, z: {output[2]:.3f}, iteration:{i}")
    print(f"position (x,y) : {my_output[0]}, z: {my_output[2][-1]:.3f}, iteration:{my_output[3]}")



