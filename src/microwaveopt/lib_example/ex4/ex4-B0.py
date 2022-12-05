import numpy as np
import pickle
import gpflow
import os
import tensorflow as tf
import math

from gpflowopt.domain import ContinuousParameter
from gpflowopt.bo import BayesianOptimizer
from gpflowopt.optim import SciPyOptimizer, StagedOptimizer, MCOptimizer
from gpflowopt.acquisition import ExpectedImprovement

from microwaveopt.momentum.design import Design
from microwaveopt.em_lib import filtering
from microwaveopt.lib_example.ex4.ex4_C_X_antenna_array import set_params

init_proj = 'ADS_Original_Files'
new_proj = 'ADS_Parametric_Layout'
antenna_array = Design(init_proj, new_proj)


def f_device(x):
    antenna_array.load_original()
    # antenna_array.layout_plot(label='True')
    ret = []
    for xi in x:
        old_geom = antenna_array.Layout.polygons
        # set variation amount for each parametrized polygon
        dims = [[0, 0] for i in range(len(old_geom))]
        dims[20][1] = xi[0]
        dims[6][1] = xi[0]
        dims[11][1] = xi[1]
        dims[9][1] = xi[1]
        dims[7][0] = xi[2]

        new_geom = set_params(old_geom, dims)
        antenna_array.Layout.overwrite_geometry(new_geom, mask='P17')

        # Updates port with the new position on the feed Polygon
        antenna_array.Ports.coordinates[0][0] = (new_geom[8].exterior.coords[0][0] + new_geom[8].exterior.coords[1][
            0]) * 1e-3 / 2
        antenna_array.Ports.coordinates[0][1] = new_geom[8].exterior.coords[3][1] * 1e-3

        # antenna_array.layout_plot(label=True)  # , coords=True)

        antenna_array.write_new()
        antenna_array.simulate_new()

        freq, s11 = antenna_array.get_s11()
        # plt.plot(freq, filtering.get_magnitude(s11))
        # plt.show()

        # insert objective function computation from s11
        ret.append(min(filtering.get_magnitude(s11, db=False)))
        #
        #
    return np.asarray(ret)[:, None]


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


r_eff = math.sqrt(1.76)            # estimated effective refraction index
operating_wavelength = 3 * 1e8 / (5.35 * 1e9) / (r_eff) *1000   # in-line wavelength in mm
width0_1 = 3.22
width0_2 = 1.55

m1_delta = width0_1 / 10
m2_delta = width0_2 / 10
m3_delta = width0_1 / 10

domain = ContinuousParameter('m0', -m1_delta, m1_delta) + \
         ContinuousParameter('m1', -m2_delta, m2_delta) + \
         ContinuousParameter('m2', -m3_delta, m3_delta) 


# Use standard Gaussian process Regression
# lhd = LatinHyperCube(10, domain)  # 21
# X = lhd.generate()
# Y = f_device(X)
# np.savetxt('X_init.txt', X)
# np.savetxt('Y_init.txt', Y)

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
r = optimizer.optimize(f_device, n_iter=50)
print(r)