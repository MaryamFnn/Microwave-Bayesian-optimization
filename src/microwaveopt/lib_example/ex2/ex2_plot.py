import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import pickle


def load_test_points(folder, space_dim):
    samples = []
    for i in [j for j in os.listdir(folder) if j.endswith('npy')]:
        curr_file = np.load(os.path.join(folder, i), allow_pickle=True)
        for k in range(curr_file.shape[0]):
            samples.append(curr_file[k])

    points = [i[0] for i in samples]
    test_x = np.asarray([np.asarray(p[:space_dim]) for p in points])
    test_y = np.asarray([np.asarray(p[space_dim:]) for p in points])

    return test_x, test_y


def feasible_region_sim(xs, ys, lb, ub):
    relevants = np.ones(shape=(xs.shape[0], 1), dtype=np.bool)
    for i in range(len(lb)):
        relevants_i = np.logical_and(ys[:, i] >= lb[i], ys[:, i] <= ub[i])
        relevants = np.logical_and(relevants, relevants_i.reshape(-1, 1))

    xf = xs[np.where(relevants)[0], :]
    yf = ys[np.where(relevants)[0], :]
    return relevants, xf, yf


def feasible_region_model(models, xs, ys, lb, ub):
    positives = np.ones(shape=(xs.shape[0], 1), dtype=np.bool)
    mu_rec = []
    for i in range(len(lb)):
        mu, var = models[i].predict_f(xs)
        mu_rec.append(mu.reshape(-1))
        positives_i = np.logical_and(mu >= lb[i], mu <= ub[i])
        positives = np.logical_and(positives, positives_i)

    xf = xs[np.where(positives)[0], :]
    yf = ys[np.where(positives)[0], :]
    return positives, xf, yf


def F1_score(positives, relevants):
    true_positives = np.logical_and(relevants, positives)
    false_positives = np.logical_and(np.invert(relevants), positives)
    false_negatives = np.logical_and(relevants, np.invert(positives))
    true_negatives = np.logical_and(np.invert(relevants), np.invert(positives))

    # deb = [np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives), np.sum(true_negatives)]
    # if sum(deb) != xs.shape[0]:
    #     raise ValueError("Error! error occurrences do not sum up to the total number of samples")

    precision = np.sum(true_positives) / np.sum(positives)
    recall = np.sum(true_positives) / np.sum(relevants)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return f1, true_positives, true_negatives, false_positives, false_negatives


def plot_sim_region(xs, relevants, name=None, debug=False):
    # ADS samples plot
    # data0 = xs[np.where(relevants)[0], :]  # non-feasible samples
    data1 = xs[np.where(relevants)[0], :]  # feasible samples

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # X = data0[:3000, 0]
    # Y = data0[:3000, 1]
    # Z = data0[:3000, 3]
    # ax.scatter(X, Y, Z, c='lightgrey', marker='o', s=2)
    X = data1[:, 0]
    Y = data1[:, 1]
    Z = data1[:, 3]
    ax.scatter(X, Y, Z, c='g', marker='o', s=7)
    ax.set_xlabel('L')
    ax.set_ylabel('S')
    ax.set_zlabel('h')
    ax.set_xlim([2, 2.5])
    ax.set_ylim([0.1, 0.2])
    ax.set_zlim([0.10, 0.15])
    ax.set_title('Simulator', fontdict={'fontsize': 15, 'fontweight': 'medium'})
    ax.view_init(elev=20., azim=-135)
    fig.tight_layout()

    if name is not None:
        fname = f"{name}.pdf"
        plt.savefig(fname)
        pickle.dump(fig, open(f"{name}.fig.pickle", 'wb'))
    if debug:
        plt.show()
    plt.clf()



def plot_model_region(xs, positives, false_positives, false_negatives, name=None, debug=False):
    # GP model samples
    data1 = xs[np.where(positives)[0], :]  # feasible samples
    data2 = xs[np.where(false_negatives)[0], :]   # misclassified samples
    data3 = xs[np.where(false_positives)[0], :]   # misclassified samples

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = data1[:, 0]
    Y = data1[:, 1]
    Z = data1[:, 3]
    ax.scatter(X, Y, Z, c='g', marker='o', s=7)
    X = data2[:, 0]
    Y = data2[:, 1]
    Z = data2[:, 3]
    ax.scatter(X, Y, Z, c='r', marker='x', s=9)
    X = data3[:, 0]
    Y = data3[:, 1]
    Z = data3[:, 3]
    ax.scatter(X, Y, Z, c='b', marker='s', s=9)
    ax.set_xlabel('L')
    ax.set_ylabel('S')
    ax.set_zlabel('h')
    ax.set_xlim([2, 2.5])
    ax.set_ylim([0.1, 0.2])
    ax.set_zlim([0.10, 0.15])
    ax.set_title('GP model', fontdict={'fontsize': 15, 'fontweight': 'medium'})
    ax.view_init(elev=20., azim=-135)
    fig.tight_layout()

    if name is not None:
        fname = f"{name}.pdf"
        plt.savefig(fname)
        pickle.dump(fig, open(f"{name}.fig.pickle", 'wb'))
    if debug:
        plt.show()
    plt.clf()


def plot_4d_rainbow(xs, ys, obj, feasibles, name=None, debug=False):
    data0 = xs[np.where(feasibles)[0], :]  # feasible samples
    data1 = ys[np.where(feasibles)[0], :]  # feasible values

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = data0[:, 0]
    Y = data0[:, 1]
    Z = data0[:, 3]
    K = data1[:, obj]
    sc = ax.scatter(X, Y, Z, c=K, cmap="viridis", marker='o', s=7)
    fig.colorbar(sc, shrink=0.5, aspect=5)
    fig.tight_layout()

    if name is not None:
        fname = f"{name}.pdf"
        plt.savefig(fname)
        pickle.dump(fig, open(f"{name}.fig.pickle", 'wb'))
    if debug:
        plt.show()
    plt.clf()















