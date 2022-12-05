import numpy as np
import os
from scipy.io import savemat
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely import affinity

from microwaveopt.momentum.design import Design
from microwaveopt.momentum.em_setup import Sampling
from microwaveopt.em_lib import filtering
from microwaveopt.em_lib import geometry as geom

# INITIAL DESIGN PARAMETERS:
width_0 = 35
length_0 = 30.5
hox_0 = 1.5
eox_0 = 4.3





def set_params(old_geometry, params):
    # assert len(old_geometry) == len(params), "ERROR! different number of polygon and specified parameters"
    # for i in params:
    #     assert len(i) == 2, f"ERROR! Number of specified dimensions is different from 2 for Polygon {i}"

    new =\
        [Polygon(p.exterior.coords) for p in old_geometry]

    # changing L2
    new[2] = geom.set_dim(new[2], params[0], 0)
    new[1] = geom.set_dim(new[1], params[0]/2, fix=1)
    new[0] = geom.set_dim(new[0], params[0]/2, fix=0)
    # new[1] = affinity.translate(new[1], xoff=-params[0]/2)
    # new[0] = affinity.translate(new[0], xoff=params[0] / 2)

    new[2] = geom.set_dim(new[2], params[1], 1, fix=0)


    # new[2] = geom.set_dim(new[2], params[0], 0)
    # new[1] = affinity.translate(new[1], xoff=params[0]/2)

    #
    # # changing S
    # new[1] = geom.set_dim(new[1], params[1]/2, 1, fix=0)
    # new[3] = geom.set_dim(new[3], params[1]/2, 1, fix=1)
    # new[2] = affinity.translate(new[2], yoff=params[1]/2)
    # new[0] = affinity.translate(new[0], yoff=-params[1]/2)

    return new


def patch_antenna(width, length, eox, hox, debug=True, sampling_mode='adaptive'):
    init_proj = 'ADS_Original_Files'
    new_proj = 'ADS_Parametric_Layout'

    device = Design(init_proj, new_proj)
    device.ads_path = "C:\\Program Files\\Keysight\\ADS2019"     # current ads path on Windows
    # device.ads_path = "/usr/local/ADS2015_01"     # current ads path on Linux

    # load initial design from init_proj folder
    device.load_original()

    # modify frequency sampling in the design class
    device.Sampling = Sampling(mode=sampling_mode, lower=2, higher=4, max_samples=50, step=0.04)

    # Modifying Layout
    # device.layout_plot(label=True, coords=True)

    old_geom = device.Layout.polygons
    # set variation amount for each parametrized polygon
    delta = [width-width_0, length-length_0]
    new_geom = set_params(old_geom, delta)
    device.Layout.overwrite_geometry(new_geom, mask='P1')

    # Modifying Ports
    # device.Ports.coordinates[0][0] = (new_geom[4].exterior.coords[0][0]) * 1e-3
    # device.Ports.coordinates[0][1] = (new_geom[4].exterior.coords[0][1] + new_geom[4].exterior.coords[3][1]) * 1e-3 / 2
    # device.Ports.coordinates[1][0] = (new_geom[4].exterior.coords[1][0]) * 1e-3
    # device.Ports.coordinates[1][1] = (new_geom[4].exterior.coords[2][1] + new_geom[4].exterior.coords[4][1]) * 1e-3 / 2
    print(device.Ports.coordinates)

    # Modifying Substrate
    device.Substrate.stack[2].height = hox * 1e-3
    device.Substrate.materials[0].permittivity = eox

    if debug:
        device.layout_plot(label=True, coords=True)

    # Save and simulate
    device.write_new()
    device.simulate_new()

    freq_step = 0
    ff_data = device.farfiels(freq_step, [1, 2], [1+0j, 1+0j], [50+0j, 50+0j])
    device.plot_farfields(ff_data, 0, 'mag', lobes=True)

    # Get frequency response
    f, s11 = device.get_s11()

    # s_matrix = s_matrix.reshape(2,2, f.shape[0])

    # bw = filtering.bandwidth(freq, transfer_mag)
    # f0 = filtering.central_frequency(freq, transfer_mag)
    # bw = round(bw / 1e9, 5)
    # f0 = round(f0 / 1e9, 5)

    if debug:
        transfer_mag = filtering.get_magnitude(s11, db=True)
        plt.plot(f, transfer_mag)
        plt.show()

    f = np.asarray(f).reshape(-1, 1)
    s11 = np.asarray(s11).reshape(-1, 1)
    s_matrix = np.array(s11)


    return f, s_matrix


n_samples = 200
S = []
freq = []
params = []
count = 0
count_f = 0

for i in range(n_samples):
    count += 1
    # # Uniform sampling
    # l2 = np.random.uniform(2, 2.5)
    # s = np.random.uniform(0.1, 0.2)
    # eox = np.random.uniform(9.5, 11)
    # hox = np.random.uniform(0.10, 0.15)

    # Gaussian sampling
    limits = np.array([[32, 38], [27.5, 33.5], [2, 5], [1, 2]])
    means = np.mean(limits, axis=1)
    devs = 0.05*means

    w = np.random.normal(means[0], devs[0])
    l = np.random.normal(means[1], devs[1])
    eox = np.random.normal(means[2], devs[2])
    hox = np.random.normal(means[3], devs[3])

    print([w, l])

    f_samp, sm = patch_antenna(w, l, eox, hox, debug=False, sampling_mode='linear')

    # end = time.time()
    # print(end - start)
    # s_db = [20*math.log10(abs(i)) for i in res.data[2][2]]
    # freq = [round(i/1e9, 6) for i in res.vars[0][3]]

    params.append([w, l, eox, hox])
    S.append(sm)
    freq.append(f_samp)
    #
    # if (bw > 7.5) & (f0 > 14.5) & (f0 < 15.5):
    #     count_f += 1


S = np.asarray(S)
freq = np.asarray(freq)

outfile = os.path.join("samples", "dataset_0.mat")
savemat(outfile, {'S': S, 'freq': freq, 'params': params})

