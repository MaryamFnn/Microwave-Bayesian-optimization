import os
import matplotlib.pyplot as plt
import microwaveopt.em_lib.filtering as filter
import numpy as np
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import box

from microwaveopt.momentum.design import Design
from microwaveopt.momentum.em_setup import Sampling

"""ORIGINAL EXAMPLE:
l1 = 4.95
l2 = 2.765
l3 = 4.95
w1 = 0.10523
w2 = 0.2097
w3 = 0.10523
d1 = 5.15
d2 = 5.15
w0 = 0.635
hox = 0.635
eox = 9.4

polygon_vertices = build_geometry([l1, l2, l3], [w1, w2, w3], [d1, d2])
"""


def run_parametric(p_vertices, debug=False):
    # DEFINE WORK FOLDERS
    ex_path = __file__
    ex_dir = os.path.abspath(os.path.join(ex_path, os.pardir))
    init_proj = os.path.join(ex_dir, "ADS_Original_Files")
    new_proj = os.path.join(ex_dir, "ADS_Parametric_Layout")

    # DEFINE DESIGN CLASS ISTANCE
    multi_stub_filter = Design(init_proj, new_proj)
    multi_stub_filter.ads_path = "C:\\Program Files\\Keysight\\ADS2019\n"

    # load initial design from init_proj folder
    multi_stub_filter.load_original()

    # modify frequency sampling in the design class
    multi_stub_filter.Sampling = Sampling(mode='linear', lower=1, higher=10, max_samples=50, step=0.1)

    # Overwrite Layout geometry (name P1) with polygons
    p_pol = [Polygon(p) for p in p_vertices]
    multi_stub_filter.Layout.overwrite_geometry(p_pol, mask='P1')

    # Update ports for the new layout
    multi_stub_filter.Ports.coordinates[0][0] = p_vertices[0][0][0] * 1e-3
    multi_stub_filter.Ports.coordinates[0][1] = p_vertices[0][3][1] * 1e-3 / 2
    multi_stub_filter.Ports.coordinates[1][0] = p_vertices[0][1][0] * 1e-3
    multi_stub_filter.Ports.coordinates[1][1] = p_vertices[0][3][1] * 1e-3 / 2

    if debug is True:
        multi_stub_filter.layout_plot()

    # write modified design in the new_proj folder
    multi_stub_filter.write_new()

    # simulate modified design from new_proj folder
    multi_stub_filter.simulate_new()

    # READ FREQUENCY RESPONSE
    freq, s21 = multi_stub_filter.get_s21()
    s21 = filter.get_magnitude(s21, db=True)

    if debug is True:
        plt.plot(freq, s21)
        plt.show()

    freq = np.asarray(freq)
    s21 = np.asarray(s21)
    return freq, s21


def build_geometry(stub_length, stub_width, stub_distance):
    """REQUIRED PARAMETERS
        stub_length : list of lenghts
        stub_width : list of widths
        stub_distance : list of reciprocal spacing, one element less than the previous lists
    """
    if (len(stub_length) == len(stub_width) == len(stub_distance) + 1) is False:
        raise ValueError("*** ERROR! Incorrect number of stub parameters***\n")

    polygons = []
    num_stubs = len(stub_length)

    pre_transmission_line = 3
    post_transmission_line = 3
    line_width = 0.635

    line_length = pre_transmission_line + sum(stub_width) + sum(stub_distance) + post_transmission_line
    # polygon defined counterclockwise from bottom left vertex
    p1 = [[0, 0], [line_length, 0], [line_length, line_width], [0, line_width]]
    polygons.append(p1)

    x0 = pre_transmission_line
    stub_distance.append(0)
    for i in range(num_stubs):
        if stub_length[i] > 0:
            new_pol = [(x0, line_width), (x0 + stub_width[i], line_width),
                       (x0 + stub_width[i], line_width + stub_length[i]), (x0, line_width + stub_length[i])]

        else:
            new_pol = [(x0, 0), (x0 + stub_width[i], 0),
                       (x0 + stub_width[i], stub_length[i]), (x0, stub_length[i])]
        x0 = x0 + stub_width[i] + stub_distance[i]
        polygons.append(new_pol)

    return polygons


def mask_distance(response, interval, value):  # , bound='high'):
    e_abs = 0
    freq = response[0]
    s_db = response[1]
    for i in range(len(freq)):
        if (freq[i] > interval[0]) & (freq[i] < interval[1]):
            e_abs += abs(value - s_db[i])
    # if bound == 'high':
    #     e_abs = e_abs
    # elif bound == 'low':
    #     e_abs = -e_abs
    # else:
    #     raise ValueError("*** ERROR! Type of bound must be either 'high' or 'low' ***\n")

    return e_abs


def convert_to_shape(v):
    X0 = np.array(v[[0], [0, 3]])
    X1 = np.array(v[1:][v[1:][:, -1, -1] > 0])[:, [0, 3, 2, 1]].reshape([-1, 2])
    X2 = np.array(v[[0], [2, 1]])
    X3 = np.flip(np.array(v[1:][v[1:][:, -1, -1] < 0])[:, [0, 3, 2, 1]].reshape([-1, 2]), axis=0)
    lr = LinearRing(np.concatenate([X0, X1, X2, X3], 0))
    return lr


if __name__ == '__main__':
    ### MC sampling
    # f_p1 = 3.5e9
    # f_s1 = 4e9
    # f_s2 = 6e9
    # f_p2 = 6.5e9
    # ret = []
    # for i in range(50):
    #     l = np.random.uniform(0, 10, 2).tolist()
    #     w = [0.625, 0.625]
    #     d = [5]
    #
    #     polygon_vertices = build_geometry(l, w, d)
    #
    #     # RUN PARAMETRIZED EXAMPLE
    #     freq, s_db = run_parametric(polygon_vertices, debug=True)
    #
    #     seg1 = s_db[np.where(freq < f_p1)]
    #     seg2 = s_db[np.where((f_s1 < freq) & (freq < f_s2))]
    #     seg3 = s_db[np.where((f_s1 < freq) & (freq < f_s2))]
    #     seg4 = s_db[np.where(freq > f_p2)]
    #
    #     term1 = np.amin(seg1)
    #     term2 = np.amax(seg2)
    #     term3 = np.amax(seg3)
    #     term4 = np.amin(seg4)
    #
    #     total_d = -term1 + term2 + term3 - term4
    #     ret.append(total_d)
    #     x = l + [total_d]
    # x = np.asarray(x)
    # np.savetxt("mc_sampling", x)




    #
    #
    l = [0., 4.89904585, 8.11868864]
    w = [0.625, 0.625, 0.625]
    d = [5, 5]


    polygon_vertices = build_geometry(l, w, d)
    #
    # RUN PARAMETRIZED EXAMPLE
    f, s = run_parametric(polygon_vertices, debug=True)

    # polygon_vertices = np.array(polygon_vertices)
    #
    # lr = convert_to_shape(polygon_vertices)  # LinearRing
    # sh = Polygon(lr)  # Polygon
    # ar = box(*list(sh.bounds)).area  # Area of bounding box
