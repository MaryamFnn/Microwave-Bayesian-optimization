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

from microwaveopt.lib_example.ex3.ex3_multi_stub_filter import run_parametric, build_geometry

f_p1 = 2e9
f_s1 = 3e9
f_s2 = 8e9
f_p2 = 9e9


def f_device(x):
    ret = []
    for xi in x:
        print(xi)
        l = xi[:3].tolist()
        w = [0.625, 0.625, 0.625]
        # d = xi[3:].tolist()
        d = [5, 5]

        polygon_vertices = build_geometry(l, w, d)
        freq, s_db = run_parametric(polygon_vertices, debug=False)
        print(freq)

        # OBECTIVE 1: STRETCH RESPONSE AT FIXED MASK
        seg1 = s_db[np.where(freq < f_p1)]
        seg2 = s_db[np.where((f_s1 < freq) & (freq < f_s2))]
        seg3 = s_db[np.where((f_s1 < freq) & (freq < f_s2))]
        seg4 = s_db[np.where(freq > f_p2)]
        term1 = np.amin(seg1)
        term2 = np.amax(seg2)
        term3 = np.amax(seg3)
        term4 = np.amin(seg4)
        total_d = -term1 + term2 + term3 - term4

        # OBECTIVE 2: ???




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


domain = ContinuousParameter('l1', 0, 10) + \
         ContinuousParameter('l2', 0, 10) + \
         ContinuousParameter('l3', 0, 10)
         # ContinuousParameter('l3', 0, 10)
         # ContinuousParameter('d1', 3, 6) + \
         # ContinuousParameter('d2', 3, 6)


domain

# Use standard Gaussian process Regression
lhd = LatinHyperCube(16, domain)  # 21
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
r = optimizer.optimize(f_device, n_iter=40)
print(r)


