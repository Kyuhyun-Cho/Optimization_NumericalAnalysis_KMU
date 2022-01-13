import matplotlib.pyplot as plt
import numpy as np

def show(function, iter_X, iter_z, iter_count):
    iter_x = np.empty(0)
    iter_y = np.empty(0)
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    X_, Y_ = np.meshgrid(x, y)
    Z_ = np.zeros((250,250))
    for i in range(250):
        for j in range(250):
            Z_[i,j] = function.get_value([X_[i,j], Y_[i,j]])
    for X in iter_X:
        iter_x = np.append(iter_x, X[0])
        iter_y = np.append(iter_y, X[1])


    anglesx = iter_x[1:] - iter_x[:-1]
    anglesy = iter_y[1:] - iter_y[:-1]

    fig = plt.figure(figsize=(16, 8))

    # Surface plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X_, Y_, Z_, rstride=5, cstride=5, cmap='jet', alpha=.4, edgecolor='none')
    ax.plot(iter_x, iter_y, iter_z, color='r', marker='*', alpha=.4)

    ax.view_init(45, 280)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Contour plot
    ax = fig.add_subplot(1, 2, 2)
    ax.contour(X_, Y_, Z_, 50, cmap='jet')
    # Plotting the iterations and intermediate values
    ax.scatter(iter_x, iter_y, color='r', marker='*')
    ax.quiver(iter_x[:-1], iter_y[:-1], anglesx, anglesy, scale_units='xy', angles='xy', scale=1, color='r', alpha=.3)
    ax.set_title('{} iterations'.format(iter_count))

    plt.show()

