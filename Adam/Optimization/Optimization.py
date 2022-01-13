import numpy as np
import copy


class Optimization:
    def __init__(self, optimizer, function, early_stop=0.00001, iter_max=10000):
        self.function = function
        self.early_stop = early_stop
        self.iter_max = iter_max

        self.optimizer_list = list()

        for i in range(self.function.dim):
            self.optimizer_list.append(copy.copy(optimizer))

    def optimize(self, init):
        val_prev = np.array(init, dtype=np.float)
        val_new = np.array(init, dtype=np.float)
        iter_X = list()
        iter_Y = list()

        iter_X.append(init)
        iter_Y.append(self.function.get_value(init))

        iter_cnt = 0
        for iter_cnt in range(self.iter_max):

            gradient = self.function.get_diff_value(val_prev)
            for i in range(len(self.optimizer_list)):
                val_new[i] = self.optimizer_list[i].update(val_prev[i], gradient[i])
            iter_X.append(val_new.copy())
            iter_Y.append(self.function.get_value(val_new))
            diff = abs(self.function.get_value(val_new) - self.function.get_value(val_prev))

            if diff < self.early_stop or np.isnan(diff):
                break
            else:
                val_prev = val_new.copy()

        return iter_X[-1], iter_X, iter_Y, iter_cnt
