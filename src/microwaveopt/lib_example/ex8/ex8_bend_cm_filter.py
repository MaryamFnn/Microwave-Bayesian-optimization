import numpy as np
import matplotlib.pyplot as plt
import os

from shapely.geometry import Polygon
from shapely import affinity
from microwaveopt.momentum.design import Design, Sampling

design_space_1 = [  # source: Yinghao Ye
    [0.1, 1],  # lower bounds
    [2, 8],  # upper bounds
]

ex_path = __file__
ex_dir = os.path.abspath(os.path.join(ex_path, os.pardir))
init_proj = os.path.join(ex_dir, "ADS_Original_Files")
new_proj = os.path.join(ex_dir, "ADS_Parametric_Layout")


def initialize():
    device = Design(init_proj, new_proj)
    # device.ads_path = "C:\Program Files\Keysight\ADS2017_Update1"  # only working with ads 2015 ¯\_(ツ)_/¯
    device.ads_path = '/usr/local/ADS2015_01'
    device.ads_licence = "27000@license.intec.ugent.be\n"
    device.load_original()
    return device


def parametrization_1(x, debug=False, afs=False):
    delta_w2 = x[0] - 0.3  # 0.6 for non-tightly coupled bend
    delta_l2 = x[1] - 8

    device = initialize()
    # device.layout_plot(label=True, coords=True)
    # plt.show()
    if afs:
        device.Sampling = Sampling(mode='linear', lower=0.2, higher=6, step=0.1)
    else:
        device.Sampling = Sampling(mode='adaptive', lower=0.2, higher=6, max_samples=50)

    geom = device.Layout.shapely()

    # Changing w2
    pol = [[[x, y] for x, y in p.exterior.coords] for p in geom]
    pol[0][0][0] -= delta_w2
    pol[0][1][0] -= delta_w2
    pol[0][8][0] -= delta_w2
    pol[0][1][1] -= delta_w2
    pol[0][2][1] -= delta_w2

    pol[1][0][0] += delta_w2
    pol[1][1][0] += delta_w2
    pol[1][8][0] += delta_w2

    pol[2][0][1] -= delta_w2
    pol[2][1][1] -= delta_w2
    pol[2][8][1] -= delta_w2
    pol[2][1][0] -= delta_w2
    pol[2][2][0] -= delta_w2

    pol[3][0][1] += delta_w2
    pol[3][1][1] += delta_w2
    pol[3][8][1] += delta_w2

    geom = [Polygon(p) for p in pol]
    geom[2] = affinity.translate(geom[2], xoff=delta_w2)
    geom[3] = affinity.translate(geom[3], xoff=delta_w2)

    # Changing l2
    pol = [[[x, y] for x, y in p.exterior.coords] for p in geom]
    pol[3][1][0] += delta_l2
    pol[3][2][0] += delta_l2
    pol[2][1][0] += delta_l2
    pol[2][2][0] += delta_l2

    pol[0][1][1] += delta_l2
    pol[0][2][1] += delta_l2
    pol[1][1][1] += delta_l2
    pol[1][2][1] += delta_l2

    geom = [Polygon(p) for p in pol]
    geom[0] = affinity.translate(geom[0], xoff=delta_l2)
    geom[1] = affinity.translate(geom[1], xoff=delta_l2)
    geom[0] = affinity.translate(geom[0], yoff=-delta_l2)
    geom[1] = affinity.translate(geom[1], yoff=-delta_l2)

    device.Layout.overwrite_geometry(geom, mask='P1')

    # # Modifying Ports
    device.Ports.coordinates[0][0] = (geom[3].exterior.coords[5][0]) * 1e-3
    device.Ports.coordinates[0][1] = (geom[3].exterior.coords[5][1] + geom[3].exterior.coords[6][1]) * 1e-3 / 2

    device.Ports.coordinates[2][0] = (geom[2].exterior.coords[5][0]) * 1e-3
    device.Ports.coordinates[2][1] = (geom[2].exterior.coords[5][1] + geom[2].exterior.coords[6][1]) * 1e-3 / 2

    device.Ports.coordinates[1][0] = (geom[1].exterior.coords[5][0] + geom[1].exterior.coords[6][0]) * 1e-3 / 2
    device.Ports.coordinates[1][1] = (geom[1].exterior.coords[5][1]) * 1e-3

    device.Ports.coordinates[3][0] = (geom[0].exterior.coords[5][0] + geom[0].exterior.coords[6][0]) * 1e-3 / 2
    device.Ports.coordinates[3][1] = (geom[0].exterior.coords[5][1]) * 1e-3

    device.Layout.overwrite_geometry(geom, mask='P1')
    if debug:
        # print(device.Ports.coordinates)
        device.layout_plot(label=True, coords=True)

    device.write_new()
    device.simulate_new(quiet=not debug)

    f, smatrix = device.get_smatrix()
    smatrix = 20 * np.log10(np.abs(smatrix))

    if debug:
        # plt.rcParams['axes.facecolor'] = 'white'
        plt.plot(f[1:], smatrix[1, 2][1:], )
        plt.plot(f[1:], smatrix[2, 2][1:], )
        plt.plot(f[1:], smatrix[3, 2][1:], )
        plt.plot(f[1:], -3 * np.ones(len(f[1:])), c='k')
        plt.plot(f[1:], -20 * np.ones(len(f[1:])), c='k')
        legend = ['s_cd21', 's_dd11', 's_dd21', '-3db', '-20db']
        plt.legend(legend)
        plt.show()

    return device


########################################################################################################################
if __name__ == '__main__':
    parametrization_1([0.1, 8], debug=True)
