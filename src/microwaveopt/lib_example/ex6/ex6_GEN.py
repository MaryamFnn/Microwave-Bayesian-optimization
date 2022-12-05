"""ORIGINAL EXAMPLE:
l1 = 4.95
l2 = 2.765
l3 = 4.95
w1 = 0.10523

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from shapely.geometry import Polygon
from shapely import affinity
from microwaveopt.momentum.design import Design
from microwaveopt.momentum.em_setup import Sampling
from microwaveopt.em_lib import filtering
from microwaveopt.em_lib import geometry as geom

def set_params(old_geometry, params):
    # assert len(old_geometry) == len(params), "ERROR! different number of polygon and specified parameters"
    # for i in params:
    #     assert len(i) == 2, f"ERROR! Number of specified dimensions is different from 2 for Polygon {i}"

    new = [Polygon(p.exterior.coords) for p in old_geometry]

    # changing l
    new[2] = affinity.translate(new[2], xoff=-params[0])
    new[3] = affinity.translate(new[3], xoff=-params[0])
    # changing g
    new[2] = affinity.translate(new[2], yoff=-params[1])
    new[3] = affinity.translate(new[3], yoff=-params[1])

    # changing w
    new[0] = geom.set_dim(new[0], params[2], dim=1, fix=0)
    new[1] = geom.set_dim(new[1], params[2], dim=0, fix=0)
    new[1] = affinity.translate(new[1], yoff=params[2])

    new[2] = geom.set_dim(new[2], params[2], dim=1, fix=1)
    new[3] = geom.set_dim(new[3], params[2], dim=0, fix=0)
    new[3] = affinity.translate(new[3], yoff=-params[2])


    return new


def tapped_line(d_l, d_g, d_w, eox, hox, debug=True, sampling_mode='adaptive'):
    d_l = round(d_l, 7)
    d_g = round(d_g, 7)
    d_w = round(d_w, 7)
    eox = round(eox, 7)
    hox = round(hox, 7)



    init_proj = 'ADS_Original_Files'
    new_proj = 'ADS_Parametric_Layout'

    device = Design(init_proj, new_proj)
    # device.ads_path = "C:\\Program Files\\Keysight\\ADS2019\n"
    # device.ads_path = "/usr/local/ADS2019"
    device.ads_path = "/usr/local/ADS2015_01"
    # device.ads_path = "/home/sumo/fgarbugl/ADS2019\n"


    # load initial design from init_proj folder
    device.load_original()

    # modify frequency sampling in the design class
    device.Sampling = Sampling(mode=sampling_mode, lower=3, higher=7, max_samples=50, step=0.1)

    if debug:
        device.layout_plot(label=True, coords=True)

    # Modifying Layout
    old_geom = device.Layout.polygons
    # set variation amount for each parametrized polygon
    delta = [d_l, d_g, d_w]
    new_geom = set_params(old_geom, delta)
    device.Layout.overwrite_geometry(new_geom, mask='P1')

    # Modifying Ports
    device.Ports.coordinates[0][0] = (new_geom[1].exterior.coords[2][0] + new_geom[1].exterior.coords[3][0]) * 1e-3 / 2
    device.Ports.coordinates[0][1] = (new_geom[1].exterior.coords[2][1]) * 1e-3
    device.Ports.coordinates[1][0] = (new_geom[3].exterior.coords[0][0] + new_geom[3].exterior.coords[1][0]) * 1e-3 / 2
    device.Ports.coordinates[1][1] = (new_geom[3].exterior.coords[0][1]) * 1e-3
    print(device.Ports.coordinates)

    if debug:
        device.layout_plot(label=True, coords=True)

    # Modifying Substrate
    device.Substrate.stack[2].height = hox * 1e-3
    device.Substrate.materials[0].permittivity = eox

    # Save and simulate
    device.write_new()
    # device.simulate_original()
    device.simulate_new()

    # Get frequency response
    f, s11 = device.get_s11()
    f, s12 = device.get_s12()
    f, s21 = device.get_s21()
    f, s22 = device.get_s22()

    if debug:
        plt.plot(f, filtering.get_magnitude(s21))
        plt.plot(f, -3*np.ones(len(f)))
        plt.plot(f, -20 * np.ones(len(f)))
        plt.show()

    f = np.asarray(f)
    s11 = np.asarray(s11)
    s12 = np.asarray(s12)
    s21 = np.asarray(s21)
    s22 = np.asarray(s22)

    linear_interp = True
    if linear_interp:
        freq_new = np.linspace(3e9, 7e9, 50)
        s11 = np.interp(freq_new, f, s11)
        s12 = np.interp(freq_new, f, s12)
        s21 = np.interp(freq_new, f, s21)
        s22 = np.interp(freq_new, f, s22)
        # plt.plot(freq_new, s_db_new)
        # plt.show()

    f = f.reshape(-1, 1)
    s11 = s11.reshape(-1, 1)
    s12 = s12.reshape(-1, 1)
    s21 = s21.reshape(-1, 1)
    s22 = s22.reshape(-1, 1)



    s_matrix = np.array([[s11, s12], [s21, s22]])
    return f, s_matrix


if __name__ == '__main__':
    f_s1 = 4e9
    f_p1 = 4.75e9
    f_p2 = 5.25e9
    f_s2 = 6e9

    l_0 = 6.977
    g_0 = 0.06
    w_0 = 1
    e_0 = 9.9
    h_0 = 0.254

    n_samples = 4
    S = []
    freq = []
    params = []
    count = 0
    count_f = 0

    for i in range(n_samples):
        count += 1

        # Gaussian sampling
        devs = 0.05
        l1 = np.random.normal(l_0, l_0*devs)
        g = np.random.normal(g_0, g_0*devs)
        w = np.random.normal(w_0, w_0*devs)
        e = np.random.normal(e_0, e_0*devs)
        h = np.random.normal(h_0, h_0*devs)

        d_l = l1 - l_0
        d_g = g - g_0
        d_w = w - w_0

        f_samp, s_db = tapped_line(d_l, d_g, d_w, e, h, debug=True, sampling_mode='linear')



        # end = time.time()
        # print(end - start)
        # s_db = [20*math.log10(abs(i)) for i in res.data[2][2]]
        # freq = [round(i/1e9, 6) for i in res.vars[0][3]]

        params.append([l1, g, w, e, h])
        S.append(s_db)
        freq.append(f_samp)

    S = np.asarray(S)
    freq = np.asarray(freq)
    outfile = os.path.join("samples", "ex6_tapped_line.mat")
    savemat(outfile, {'S': S, 'freq': freq, 'params': params})

