import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from shapely.geometry import Polygon
from shapely import affinity
from microwaveopt.momentum.design import Design
from microwaveopt.momentum.em_setup import Sampling
from microwaveopt.em_lib import filtering
from microwaveopt.em_lib import geometry as geom

design_space_1 = [  # source: Yinghao Ye
    [0.3, 0.8, 19.1],  # lower bounds
    [0.5, 1.0, 19.3],  # upper bounds
]

# parent = os.path.abspath(os.path.join(__file__, os.pardir))
# init_proj = os.path.join(parent, 'ADS_Original_Files')
# new_proj = os.path.join(parent, 'ADS_Parametric_Layout')

init_proj = r"C:\\Users\Administrator\Desktop\Sources\\MicrowaveOptProject\\MyCode forZigzagFilter\\MicrowaveOpt-new\\microwaveopt\\lib_example\\ex7\ADS_Original_Files"
new_proj = r"C:\\Users\Administrator\Desktop\Sources\\MicrowaveOptProject\\MyCode forZigzagFilter\\MicrowaveOpt-new\\microwaveopt\\lib_example\\ex7\ADS_Parametric_Layout"


def initialize():
    device = Design(init_proj, new_proj)
    device.ads_path = "/usr/local/ADS2015_01"  # only working with ads 2015 ¯\_(ツ)_/¯
    device.ads_licence = "27000@license.intec.ugent.be\n"
    device.load_original()
    return device


def parametrization_1(x, debug=False, afs=False, step=0.05):
    delta_G = 0  # x[0] - 0.3
    delta_D = x[0] - 0.8
    delta_L = x[1] - 18

    device = initialize()
    # device.layout_plot(label=True, coords=True)
    if afs:
        device.Sampling = Sampling(mode='adaptive', lower=1, higher=4.5, max_samples=50)
    else:
        device.Sampling = Sampling(mode='linear', lower=1, higher=4.5, step=step)


    old_geom = device.Layout.polygons
    pol = [Polygon(p.shapely.exterior.coords) for p in old_geom]

    # Changing G
    pol[1] = affinity.translate(pol[1], yoff=-delta_G)
    pol[17] = affinity.translate(pol[17], yoff=-delta_G)
    pol[2] = affinity.translate(pol[2], yoff=-delta_G)
    pol[13] = affinity.translate(pol[13], yoff=delta_G)
    pol[14] = affinity.translate(pol[14], yoff=delta_G)
    pol[18] = affinity.translate(pol[18], yoff=delta_G)

    # Changing D
    pol[7] = affinity.translate(pol[7], xoff=delta_D)
    pol[5] = affinity.translate(pol[5], xoff=delta_D)
    pol[10] = affinity.translate(pol[10], xoff=delta_D)
    pol[11] = affinity.translate(pol[11], xoff=delta_D)
    pol[15] = affinity.translate(pol[15], xoff=delta_D)
    pol[14] = affinity.translate(pol[14], xoff=delta_D)
    pol[12] = affinity.translate(pol[12], xoff=delta_D)
    pol[13] = affinity.translate(pol[13], xoff=delta_D)
    pol[19] = affinity.translate(pol[19], xoff=delta_D)
    pol[18] = affinity.translate(pol[18], xoff=delta_D)

    # Changing L
    pol[5] = geom.set_dim(pol[5], delta_L, dim=1, fix=0)
    pol[6] = geom.set_dim(pol[6], delta_L, dim=1, fix=0)
    pol[9] = affinity.translate(pol[9], yoff=delta_L)
    pol[10] = affinity.translate(pol[10], yoff=delta_L)
    pol[11] = affinity.translate(pol[11], yoff=delta_L)
    pol[14] = affinity.translate(pol[14], yoff=delta_L)
    pol[15] = affinity.translate(pol[15], yoff=delta_L)
    pol[13] = affinity.translate(pol[13], yoff=delta_L)
    pol[12] = affinity.translate(pol[12], yoff=delta_L)
    pol[18] = affinity.translate(pol[18], yoff=delta_L)
    pol[19] = affinity.translate(pol[19], yoff=delta_L)

    # Modifying Ports
    device.Ports.coordinates[0][0] = (pol[17].exterior.coords[0][0]) * 1e-3
    device.Ports.coordinates[0][1] = (pol[17].exterior.coords[0][1] + pol[17].exterior.coords[3][1]) * 1e-3 / 2

    device.Ports.coordinates[1][0] = (pol[18].exterior.coords[1][0]) * 1e-3
    device.Ports.coordinates[1][1] = (pol[18].exterior.coords[1][1] + pol[18].exterior.coords[2][1]) * 1e-3 / 2

    device.Layout.overwrite_geometry(pol, mask='P1')

    quiet = True
    if debug:
        print(device.Ports.coordinates)
        device.layout_plot(label=True, coords=True)
        quiet = False

    device.write_new()
    device.simulate_new(quiet=quiet)

    return device


########################################################################################################################

if __name__ == '__main__':
    init_proj = r"C:\\Users\Administrator\Desktop\Sources\\MicrowaveOptProject\\MyCode forZigzagFilter\\MicrowaveOpt-new\\microwaveopt\\lib_example\\ex7\ADS_Original_Files"
    new_proj = r"C:\\Users\Administrator\Desktop\Sources\\MicrowaveOptProject\\MyCode forZigzagFilter\\MicrowaveOpt-new\\microwaveopt\\lib_example\\ex7\ADS_Parametric_Layout"


    # init_proj = 'ADS_Original_Files'
    # new_proj = 'ADS_Parametric_Layout'
    #
    # params = [1, 21]
    # saved_device = f'device_obj_{params}.pkl'
    # if os.path.exists(saved_device):
    #     with open(saved_device, 'rb') as file:
    #         device = pickle.load(file)
    # else:
    #     device = parametrization_1(params, debug=True)
    #     with open(saved_device, 'wb') as file:
    #         pickle.dump(device, file)
    #
    # device.layout_plot(label=True, coords=True)
    #
    # f, smatrix = device.get_smatrix()
    # smatrix_db = 20 * np.log10(np.abs(smatrix))
    #
    # # p_loss = 20 * np.log10(1 - np.abs(smatrix[0, 0]) ** 2 - np.abs(smatrix[1, 0]) ** 2)
    #
    #
    #
    # f = np.asarray(f)/1e9
    # y = np.abs(smatrix[0, 1])
    # BW = np.array([2.45, 2.55])
    # LBW_idx = (f < BW[0])
    # BW_idx = np.bitwise_and((f >= BW[0]), (f <= BW[1])).flatten()
    # HBW_idx = (f > BW[1])
    # bw_max_gain = 10 ** (-3 / 20)
    # off_min_gain = 10 ** (-40 / 20)
    #
    # y = np.abs(y).reshape(1, -1)
    # # Uncomment to apply bandpass mask
    # y_lower_band = y[:, LBW_idx]
    # y_inband = y[:, BW_idx]
    # y_higher_band = y[:, HBW_idx]
    #
    # y_new = np.concatenate(
    #     (
    #         (off_min_gain - y_lower_band) / len(y_lower_band),
    #         (y_inband - bw_max_gain) / len(y_inband),
    #         (off_min_gain - y_higher_band) / len(y_higher_band)
    #     ),
    #     axis=-1
    # )
    #
    # plt.rcParams['text.usetex'] = True
    # fig = plt.figure(figsize=[5,4])
    # # plt.plot(f, smatrix[0, 0])
    # plt.plot(f, np.abs(smatrix[0, 1]))
    # plt.plot(f, y_new.flatten())
    #
    # plt.plot(f[28:33], 0.708 * np.ones(len(f[28:33])))
    # plt.plot(f[:29], 0.01 * np.ones(len(f[:29])), c='g')
    # plt.plot(f[32:], 0.01 * np.ones(len(f[32:])), c='g')
    # plt.vlines(2.40, 0.01, 0.708, colors='g')
    # plt.vlines(2.60, 0.01, 0.708, colors='g')
    #
    # legend = [r'$|s_{21}(\textbf{p},f)|$', r'$g(s_{21},f)$', 'specifications \n threshold']
    # plt.legend(legend, fontsize=16, loc='lower right')
    # plt.xlabel('frequency [GHz]', fontsize=16)
    # plt.ylabel('mag', loc='top', rotation='0', fontsize=16)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # # plt.title(r'Simulated response for \textbf{p}=[1, 21]', fontsize=14)
    # # plt.ylim([-100, 0])
    # plt.show()



    params = [0.3, 25]
    # device = parametrization_1(params, debug=True)
    saved_device = f'device_obj_{params}.pkl'
    # with open(saved_device, 'wb') as file:
    #     pickle.dump(device, file)
    with open(saved_device, 'rb') as file:
        device = pickle.load(file)
    f, smatrix = device.get_smatrix()
    f = np.asarray(f) / 1e9

    # params = [1.16, 18.75]
    # device = parametrization_1(params, debug=True)
    # saved_device = f'device_obj_{params}.pkl'
    # with open(saved_device, 'wb') as file:
    #     pickle.dump(device, file)
    # with open(saved_device, 'rb') as file:
    #     device = pickle.load(file)
    # f2, smatrix2 = device.get_smatrix()

    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=[5,4])
    # plt.plot(f, smatrix[0, 0])
    plt.plot(f, 20*np.log10(np.abs(smatrix[0, 1])))
    # plt.plot(f, np.abs(smatrix2[0, 1]))

    plt.hlines(-40, 1, 2.45, colors='g')
    plt.hlines(-3, 2.45, 2.55, colors='g')
    plt.hlines(-40, 2.55, 4.5, colors='g')
    plt.vlines(2.45, -40, -3, colors='g')
    plt.vlines(2.55, -40, -3, colors='g')

    # legend = [r'BO: $\textbf{p}_{opt} = [1.14, 17.25]$', r'DB-BO: $\textbf{p}_{opt} = [1.16, 17.37]$', r'worst-case $s_{21}$']
    legend = [r'Response for L=25, G=0.3', r'Desired specifications']
    plt.legend(legend, fontsize=14, loc='upper right')
    plt.xlabel('frequency [GHz]', fontsize=16, )
    plt.ylabel(r'$|s_{21}| [dB] \qquad {\,}$   ',  fontsize=16, rotation=0, loc='top')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title(r'Optimal response', fontsize=14)
    # plt.ylim([-100, 0])
    plt.show()