import numpy as np
import pickle
import gpflow
import os
import tensorflow as tf

from gpflowopt.domain import ContinuousParameter
from gpflowopt.bo import BayesianOptimizer
from gpflowopt.design import LatinHyperCube
from gpflowopt.optim import SciPyOptimizer, StagedOptimizer, MCOptimizer
from gpflowopt.acquisition import ExpectedImprovement

from microwaveopt.lib_example.ex6.ex6_tapped_line import parametrization_1

l_0 = 6.977
g_0 = 0.06
w_0 = 1
e_0 = 9.9
h_0 = 0.254

f_s1 = 4e9
f_p1 = 4.75e9
f_p2 = 5.25e9
f_s2 = 6e9


def f_device(x):
    ret = []
    for xi in x:
        print(xi)
        d_l = xi[0] - l_0
        d_g = xi[1] - g_0
        # d_w = xi[2] - w_0
        d_w = 0
        e = xi[2] # xi[3]
        h = xi[3] # xi[4]

        freq, s_db = parametrization_1(d_l, d_g, d_w, e, h, debug=False, sampling_mode='adaptive')

        # OBECTIVE 1: STRETCH RESPONSE AT FIXED MASK
        seg1 = s_db[np.where(freq < f_s1)]
        seg2 = s_db[np.where((f_p1 < freq) & (freq < f_p2))]
        seg3 = s_db[np.where((f_p1 < freq) & (freq < f_p2))]
        seg4 = s_db[np.where(freq > f_s2)]
        term1 = np.amax(seg1)
        term2 = np.amin(seg2)
        term3 = np.amin(seg3)
        term4 = np.amax(seg4)

        total_d = term1 - 2*term2 - 2*term3 + term4

        # CHEAP OBJECTIVE: footprint
        footprint = (xi[0] + 10) * (4 * 2 + xi[1])

        ret.append(total_d)  # negate for maximization

    return np.asarray(ret)[:, None]


# polygon_vertices = build_geometry([5, 6, 7], [0.625, 0.625, 0.625], [5, 5])
# get_SBF3(polygon_vertices, debug=1, sampling_mode='adaptive', interval=[1, 10])


def callb(m):
    it[0] += 1
    f_name = os.path.join('models', f"model_{it[0]}")
    with open(f_name, 'wb') as model_file:
        pickle.dump(m, model_file, pickle.HIGHEST_PROTOCOL)

    if it[0] % 1 == 0:
        # save sequentially samples points
        np.savetxt('Xs_sofar.txt', alpha.data[0])
        np.savetxt('Ys_sofar.txt', alpha.data[1])
    tf.reset_default_graph()


domain = ContinuousParameter('l', 4., 10.) + \
         ContinuousParameter('g', 0.0, 0.1) + \
         ContinuousParameter('e', 8, 11)  + \
         ContinuousParameter('h', 0.2, 0.4)
# ContinuousParameter('w', 0.8, 1.2) +\

domain

# Use standard Gaussian process Regression
lhd = LatinHyperCube(20, domain)  # 21
X = lhd.generate()
Y = f_device(X)
np.savetxt('X_init.txt', X)
np.savetxt('Y_init.txt', Y)

X = np.loadtxt('X_init.txt')
Y = np.loadtxt('Y_init.txt')

Y = Y.reshape(-1, 1)

model = gpflow.gpr.GPR(X.copy(), Y.copy(), gpflow.kernels.Matern52(2, ARD=True))
model.kern.lengthscales.transform = gpflow.transforms.Log1pe(1e-3)

# Now create the Bayesian Optimizer
alpha = ExpectedImprovement(model)

acquisition_opt = StagedOptimizer([MCOptimizer(domain, 200),
                                   SciPyOptimizer(domain)])

it = [0]

optimizer = BayesianOptimizer(domain, alpha, optimizer=acquisition_opt, callback=callb, verbose=True)

# Run the Bayesian optimization
r = optimizer.optimize(f_device, n_iter=80)
print(r)
