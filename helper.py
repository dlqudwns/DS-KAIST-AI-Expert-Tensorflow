from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def initialize_plot(cost_func):
    # pyplot settings
    plt.ion()
    fig = plt.figure(figsize=(3, 2), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    params = {'legend.fontsize': 3,
              'legend.handlelength': 3}
    plt.rcParams.update(params)
    plt.axis('off')

    # input (x, y) and output (z) nodes of cost-function graph
    x, y, z = cost_func()

    # visualize cost function as a contour plot
    x_val = y_val = np.arange(-1.5, 1.5, 0.005, dtype=np.float32)
    x_val_mesh, y_val_mesh = np.meshgrid(x_val, y_val)
    x_val_mesh_flat = x_val_mesh.reshape([-1, 1])
    y_val_mesh_flat = y_val_mesh.reshape([-1, 1])
    with tf.Session() as sess:
        z_val_mesh_flat = sess.run(z, feed_dict={x: x_val_mesh_flat, y: y_val_mesh_flat})
    z_val_mesh = z_val_mesh_flat.reshape(x_val_mesh.shape)
    levels = np.arange(-10, 1, 0.05)
    ax.plot_surface(x_val_mesh, y_val_mesh, z_val_mesh, alpha=.4, cmap=cm.coolwarm)
    plt.draw()

    # 3d plot camera zoom, angle
    xlm = ax.get_xlim3d()
    ylm = ax.get_ylim3d()
    zlm = ax.get_zlim3d()
    ax.set_xlim3d(xlm[0] * 0.5, xlm[1] * 0.5)
    ax.set_ylim3d(ylm[0] * 0.5, ylm[1] * 0.5)
    ax.set_zlim3d(zlm[0] * 0.5, zlm[1] * 0.5)
    azm = ax.azim
    ele = ax.elev + 40
    ax.view_init(elev=ele, azim=azm)

    return ax

