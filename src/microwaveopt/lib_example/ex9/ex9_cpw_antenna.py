import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from shapely.geometry import Polygon
from shapely import affinity
from microwaveopt.momentum.design import Design, Sampling
import microwaveopt.em_lib.geometry as geom

design_space_1 = [  # source: Koziel
    [29, 5, 17, 0.2, 1.5, 0.5],  # lower bounds
    [42, 12, 25, 0.6, 5.2, 3.5],  # upper bounds
]


parent = os.path.abspath(os.path.join(__file__, os.pardir))
init_proj = os.path.join(parent, 'ADS_Original_Files')
new_proj = os.path.join(parent, 'ADS_Parametric_Layout')

def initialize():
    device = Design(init_proj, new_proj)
    # device.ads_path = "C:\Program Files\Keysight\ADS2017_Update1"  # only working with ads 2015 ¯\_(ツ)_/¯
    # device.ads_path = '/usr/local/ADS2015_01'
    device.ads_licence = "27000@license.intec.ugent.be\n"
    device.load_original()
    return device


def parametrization_1(x, debug=False, afs=False, step=0.09):
    delta_l1 = x[0] - 29
    delta_l2 = x[1] - 5
    delta_l3 = x[2] - 17
    delta_w1 = x[3] - 0.2
    delta_w2 = x[4] - 1.5
    delta_w3 = x[5] - 0.5

    device = initialize()
    if afs:
        device.Sampling = Sampling(mode='adaptive', lower=1.5, higher=6, max_samples=51)
    else:
        device.Sampling = Sampling(mode='linear', lower=1.5, higher=6, step=step)

    # if debug:
    #     device.layout_plot(label=True, coords=True)
    #     plt.show()

    mask1 = device.Layout.shapely(mask='P1')
    mask3 = device.Layout.shapely(mask='P3')

    # l1
    mask1[0] = geom.set_dim(mask1[0], 2*delta_l1, dim=0)
    mask3[5] = geom.set_dim(mask3[5], delta_l1, dim=0, fix=1)
    mask3[1] = geom.set_dim(mask3[1], delta_l1, dim=0, fix=0)
    # l2
    mask3[4] = geom.set_dim(mask3[4], delta_l2, dim=0, fix=1)
    mask3[2] = geom.set_dim(mask3[2], delta_l2, dim=0, fix=0)
    # l2
    mask3[6] = geom.set_dim(mask3[6], delta_l3, dim=0, fix=1)
    mask3[3] = geom.set_dim(mask3[3], delta_l3, dim=0, fix=0)
    # w1
    mask3[5] = geom.set_dim(mask3[5], delta_w1, dim=1, fix=0)
    mask3[1] = geom.set_dim(mask3[1], delta_w1, dim=1, fix=0)
    mask3[4] = affinity.translate(mask3[4], yoff=delta_w1)
    mask3[2] = affinity.translate(mask3[2], yoff=delta_w1)
    mask3[6] = affinity.translate(mask3[6], yoff=delta_w1)
    mask3[3] = affinity.translate(mask3[3], yoff=delta_w1)
    # w2
    mask3[4] = geom.set_dim(mask3[4], delta_w2, dim=1, fix=0)
    mask3[2] = geom.set_dim(mask3[2], delta_w2, dim=1, fix=0)
    mask3[6] = affinity.translate(mask3[6], yoff=delta_w2)
    mask3[3] = affinity.translate(mask3[3], yoff=delta_w2)
    # w3
    mask3[6] = geom.set_dim(mask3[6], delta_w3, dim=1, fix=0)
    mask3[3] = geom.set_dim(mask3[3], delta_w3, dim=1, fix=0)

    mask1[0] = geom.set_dim(mask1[0], delta_w1+delta_w3+delta_w3, dim=1, fix=0)

    # #g
    # mask3[0] = geom.set_dim(mask3[0], 0.15, dim=0, fix=0)
    # mask3[7] = geom.set_dim(mask3[7], 0.15, dim=0, fix=1)
    #
    # device.Substrate.materials[0].permittivity = 4.3
    # device.Substrate.stack[2].height = 0.0016


    device.Layout.overwrite_geometry(mask1, mask='P1')
    device.Layout.overwrite_geometry(mask3, mask='P3')

    if debug:
        # print(device.Ports.coordinates)
        device.layout_plot(label=True, coords=True)

    device.write_new()
    device.simulate_new(quiet=not debug)

    f, smatrix = device.get_smatrix_diff()
    smatrix = 20 * np.log10(np.abs(smatrix))

    if debug:
        # plt.rcParams['axes.facecolor'] = 'white'
        plt.plot(f[1:], smatrix[1:,0], )
        plt.plot(f[1:], -3 * np.ones(len(f[1:])), c='k')
        plt.plot(f[1:], -20 * np.ones(len(f[1:])), c='k')
        legend = ['s11', '-3db', '-20db']
        plt.legend(legend)
        plt.show()

    return device


########################################################################################################################
if __name__ == '__main__':
    # orig_opt = [31, 10, 17, 0.3, 4, 1]
    # orig_opt = [36.34129349, 11.97609119, 23.23653667 , 0.2, 1.5, 0.5]

    # params = [34.93434417, 10.7444943,  20.37140868, 0.2, 1.5, 0.5]
    params = [34.25213061, 11.42128616, 22.41028743, 0.2, 1.5, 0.5]
    device = parametrization_1(params, debug=True)
    saved_device = f'device_obj_{params}.pkl'
    with open(saved_device, 'wb') as file:
        pickle.dump(device, file)
    with open(saved_device, 'rb') as file:
        device = pickle.load(file)
    f, smatrix = device.get_smatrix_diff()
    f = np.asarray(f) / 1e9

    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=[5,4])
    # plt.plot(f, smatrix[0, 0])
    plt.plot(f, 20*np.log10(np.abs(smatrix)))
    # plt.plot(f, y_new.flatten())

    plt.hlines(-3, 1.5, 1.9, colors='g')
    plt.hlines(-20, 2.2, 2.7, colors='g')
    plt.hlines(-3, 3, 4, colors='g')
    plt.hlines(-20, 4.3, 4.8, colors='g')
    plt.hlines(-3, 5.1, 6, colors='g')
    # plt.vlines(2.3, -20, -3, colors='g')
    # plt.vlines(2.7, -20, -3, colors='g')
    # plt.vlines(4.8, -20, -3, colors='g')
    # plt.vlines(5.2, -20, -3, colors='g')

    legend = [r'$|s_{21}(\textbf{p},f)|$', r'$g(s_{21},f)$', 'specifications \n threshold']
    plt.legend(legend, fontsize=16, loc='lower left')
    plt.xlabel('frequency [GHz]', fontsize=16)
    plt.ylabel('mag', loc='top', rotation='0', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title(r'Simulated response for \textbf{p}=[1, 21]', fontsize=14)
    # plt.ylim([-100, 0])
    plt.show()

