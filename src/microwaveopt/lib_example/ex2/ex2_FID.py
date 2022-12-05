import numpy as np
import os
import tensorflow as tf
import pickle
import time
import matplotlib.pyplot as plt

import gpflow
import gpflowopt
from gpflowopt.bo import BayesianOptimizer
from gpflowextra.acquisition.entropy_feasible import Multi_Model_Entropy_Feasible


from shapely.geometry import Polygon
from shapely import affinity
from microwaveopt.momentum.design import Design
from microwaveopt.momentum.em_setup import Sampling
from microwaveopt.em_lib import filtering
from microwaveopt.em_lib import geometry as geom

from microwaveopt.lib_example.ex2 import ex2_plot

# INITIAL DESIGN PARAMETERS:
w1_0 = 0.1185
w2_0 = 0.1185
l2_0 = 2.1946
l1_0 = l2_0 - 2 * w2_0
l3_0 = 0.2
s_0 = 0.1219
hox_0 = 0.127
eox_0 = 9.9


def set_params(old_geometry, params):
    # assert len(old_geometry) == len(params), "ERROR! different number of polygon and specified parameters"
    # for i in params:
    #     assert len(i) == 2, f"ERROR! Number of specified dimensions is different from 2 for Polygon {i}"

    new = [Polygon(p.exterior.coords) for p in old_geometry]

    # changing L2
    new[4] = geom.set_dim(new[4], params[0], 0)
    new[0] = geom.set_dim(new[0], params[0], 0)
    new[2] = geom.set_dim(new[2], params[0], 0)
    new[1] = affinity.translate(new[1], xoff=params[0]/2)
    new[3] = affinity.translate(new[3], xoff=-params[0]/2)
    #
    # # changing S
    new[1] = geom.set_dim(new[1], params[1]/2, 1, fix=0)
    new[3] = geom.set_dim(new[3], params[1]/2, 1, fix=1)
    new[2] = affinity.translate(new[2], yoff=params[1]/2)
    new[0] = affinity.translate(new[0], yoff=-params[1]/2)

    return new


def folded_stub_filter(l2, s, eox, hox, debug=True, sampling_mode='adaptive'):
    init_proj = 'ADS_Original_Files'
    new_proj = 'ADS_Parametric_Layout'

    device = Design(init_proj, new_proj)
    device.ads_path = "C:\\Program Files\\Keysight\\ADS2015_01\n"

    # load initial design from init_proj folder
    device.load_original()

    # modify frequency sampling in the design class
    device.Sampling = Sampling(mode=sampling_mode, lower=7, higher=21, max_samples=50, step=0.2)

    # Modifying Layout
    old_geom = device.Layout.polygons
    # set variation amount for each parametrized polygon
    delta = [l2-l2_0, s-s_0]
    new_geom = set_params(old_geom, delta)
    device.Layout.overwrite_geometry(new_geom, mask='P1')

    # Modifying Ports
    device.Ports.coordinates[0][0] = (new_geom[4].exterior.coords[0][0]) * 1e-3
    device.Ports.coordinates[0][1] = (new_geom[4].exterior.coords[0][1] + new_geom[4].exterior.coords[3][1]) * 1e-3 / 2
    device.Ports.coordinates[1][0] = (new_geom[4].exterior.coords[1][0]) * 1e-3
    device.Ports.coordinates[1][1] = (new_geom[4].exterior.coords[2][1] + new_geom[4].exterior.coords[4][1]) * 1e-3 / 2
    print(device.Ports.coordinates)

    # Modifying Substrate
    device.Substrate.stack[2].height = hox * 1e-3
    device.Substrate.materials[0].permittivity = eox

    if debug:
        device.layout_plot(label=True, coords=True)

    # Save and simulate
    device.write_new()
    device.simulate_new()

    # Get frequency response
    freq, s21 = device.get_s21()
    transfer_mag = filtering.get_magnitude(s21)
    s21 = transfer_mag
    bw = filtering.bandwidth(freq, transfer_mag)
    f0 = filtering.central_frequency(freq, transfer_mag)
    bw = round(bw / 1e9, 5)
    f0 = round(f0 / 1e9, 5)

    if debug:
        plt.plot(freq, transfer_mag)
        plt.show()

    return bw, f0, freq, s21
init_proj = 'ADS_Original_Files'
new_proj = 'ADS_Parametric_Layout'


def f(x):
    """
    :param x: [L, S, h_ox, e_ox]
    :return:
    """
    x = x.flatten()
    bw, f0, _, _ = folded_stub_filter(float(x[0]), float(x[1]), float(x[2]), float(x[3]), debug=False)
    ret = [bw, f0]
    return np.atleast_2d(np.asarray(ret))


MAX_ITERATIONS = 40
Q = 4
N = 10  # Size of the initial design
f1_bounds = [7, 7.1]
f2_bounds = [13.5, 14]
bounds = np.array([[2, 2.5], [0.1, 0.2], [9.5, 11], [0.10, 0.15]])
lb = bounds[:, 0]
ub = bounds[:, 1]
d = np.sum([gpflowopt.domain.ContinuousParameter('x' + str(i), lb[i], ub[i]) for i in range(Q)])
domain = d

run_dir = os.path.join('logs', f"run_{int(time.time())}")
os.mkdir(run_dir)
model_dir = os.path.join(run_dir, 'models')
os.mkdir(model_dir)

# X_init = LHS(N, d).generate()
# np.savetxt(os.path.join("logs", "run_00", 'X_init.txt'), X_init)
# Y_init = np.zeros(shape=(N, 2))
# for i in range(N):
#     print(X_init[i])
#     Y_init[i, :] = f(X_init[i])
#     np.savetxt(os.path.join("logs", "run_00", 'Y_init.txt'), Y_init)

X_init = np.loadtxt(os.path.join('logs', 'run_00', 'X_init.txt'))
Y_init = np.loadtxt(os.path.join('logs', 'run_00', 'Y_init.txt'))

# One model for each objective
objective_models = [gpflow.gpr.GPR(X_init.copy(), Y_init[:, [i]].copy(), gpflow.kernels.Matern32(Q, ARD=True)) for i in
                    range(Y_init.shape[1])]
for model in objective_models:
    model.likelihood.variance = 1e-6

# The acquisition model which depends on the model is constructed here
# a = ES(m, d, high, low) # model, dimension, high_bound, low_bound
#
acq_func = Multi_Model_Entropy_Feasible(model=objective_models, domain=domain, f_boundaries=[[f1_bounds], [
        f2_bounds]])  # model, dimension, high_bound, low_bound
it = []
average_CS = []
imse_score = []
f1_score = []


# Plotting Real Feasible region
test_x, test_y = ex2_plot.load_test_points('samples', Q)
relevants, xf_sim, yf_sim = ex2_plot.feasible_region_sim(test_x, test_y,
                                                         [f1_bounds[0], f2_bounds[0]], [f1_bounds[1], f2_bounds[1]])
ex2_plot.plot_sim_region(test_x, relevants, name=os.path.join(run_dir, "test_region"), debug=True)
ex2_plot.plot_4d_rainbow(test_x, test_y, 0, relevants, name=os.path.join(run_dir, "test_region_obj1"), debug=True)    # rainbow plot for objective 1
ex2_plot.plot_4d_rainbow(test_x, test_y, 1, relevants, name=os.path.join(run_dir, "test_region_obj2"), debug=True)    # rainbow plot for objective 2


# One model for each objective
def callb(models):
    it.append(1)
    for i in range(len(models)):
        f_name = os.path.join(model_dir, f"model_{len(it)}_{i}")
        with open(f_name, 'wb') as model_file:
            # Step 3
            pickle.dump(models[i], model_file, pickle.HIGHEST_PROTOCOL)
    tf.reset_default_graph()

    if len(it) % 1 == 0:
        # save sequentially samples points
        np.savetxt(os.path.join(run_dir, 'Xs_sofar.txt'), acq_func.data[0])
        np.savetxt(os.path.join(run_dir, 'Ys_sofar.txt'), acq_func.data[1])

        relevants, _, _ = ex2_plot.feasible_region_sim(test_x, test_y,
                                                       [f1_bounds[0], f2_bounds[0]],
                                                       [f1_bounds[1], f2_bounds[1]])
        positives, _, _ = ex2_plot.feasible_region_model(models, test_x, test_y,
                                                         [f1_bounds[0], f2_bounds[0]],
                                                         [f1_bounds[1], f2_bounds[1]])

        f1, tp, tn, fp, fn = ex2_plot.F1_score(positives, relevants)

        f1_score.append(f1)
        np.savetxt(os.path.join(run_dir, f"f1.txt"), np.atleast_2d(np.asarray(f1_score)))

        ex2_plot.plot_model_region(test_x, positives, fp, fn, name=os.path.join(run_dir, f"f_region{len(it)}"), debug=False)
        
    if len(it) == MAX_ITERATIONS:
        np.savetxt(os.path.join(run_dir, f"false_positives_{len(it)}.txt"), np.atleast_2d(fp))
        np.savetxt(os.path.join(run_dir, f"false_negatives_{len(it)}.txt"), np.atleast_2d(fn))

        # plot feasible region for objective 1
        relevants_1, _, _ = ex2_plot.feasible_region_sim(test_x, test_y, [f1_bounds[0]], [f1_bounds[1]])
        positives_1, _, _ = ex2_plot.feasible_region_model([models[0]], test_x, test_y, [f1_bounds[0]], [f1_bounds[1]])
        f1_obj1, tp_1, tn_1, fp_1, fn_1 = ex2_plot.F1_score(positives_1, relevants_1)
        ex2_plot.plot_model_region(test_x, positives_1, fp_1, fn_1, name=os.path.join(run_dir, f"obj1_region"), debug=False)

        # plot feasible region for objective 1
        relevants_2, _, _ = ex2_plot.feasible_region_sim(test_x, test_y, [f2_bounds[0]], [f2_bounds[1]])
        positives_2, _, _ = ex2_plot.feasible_region_model([models[1]], test_x, test_y, [f2_bounds[0]], [f2_bounds[1]])
        f2_obj2, tp_2, tn_2, fp_2, fn_2 = ex2_plot.F1_score(positives_2, relevants_2)
        ex2_plot.plot_model_region(test_x, positives_2, fp_2, fn_2, name=os.path.join(run_dir, f"obj2_region"), debug=False)




# Here the optimizer for the acquisition is chosen. 1000 points are chosen at random,
# the best point is selected and starting from that point a gradient optimizer is used.
acquisition_opt = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(domain, 1000),
                                                   gpflowopt.optim.SciPyOptimizer(domain)])
# acquisition optimizer
optimizer = BayesianOptimizer(domain, acq_func, optimizer=acquisition_opt, scaling=True, callback=callb,
                              verbose=False)

# 50 iterations are performed
result = optimizer.optimize([f], n_iter=MAX_ITERATIONS)  # pp.f_n 是真实带噪声函数
