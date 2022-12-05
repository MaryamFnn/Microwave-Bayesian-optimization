import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import box
from shapely.ops import cascaded_union

from microwaveopt.momentum.design import Design
from microwaveopt.momentum.em_setup import Sampling
import microwaveopt.em_lib.filtering as filtering

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

layout_polygons = build_geometry([l1, l2, l3], [w1, w2, w3], [d1, d2])
"""


def run_parametric(pol_list, debug=False):
    # DEFINE WORK FOLDERS
    ex_path = __file__
    ex_dir = os.path.abspath(os.path.join(ex_path, os.pardir))
    init_proj = os.path.join(ex_dir, "ADS_Original_Files")
    new_proj = os.path.join(ex_dir, "ADS_Parametric_Layout")

    # DEFINE DESIGN CLASS ISTANCE
    multi_stub_filter = Design(init_proj, new_proj)
    multi_stub_filter.ads_path = "C:\\Program Files\\Keysight\\ADS2015_01\n"

    # load initial design from init_proj folder
    multi_stub_filter.load_original()

    if debug is True:
        multi_stub_filter.layout_plot()

    # modify frequency sampling in the design class
    multi_stub_filter.Sampling = Sampling(mode='adaptive', lower=1, higher=10, max_samples=50)

    # Overwrite Layout geometry (name P1) with polygons
    multi_stub_filter.Layout.overwrite_geometry(pol_list, mask='P1')

    # Update ports for the new layout, the first polygon of the list is the main transmission line
    multi_stub_filter.Ports.coordinates[0][0] = pol_list[0].exterior.coords[0][0] * 1e-3
    multi_stub_filter.Ports.coordinates[0][1] = pol_list[0].exterior.coords[3][1] * 1e-3 / 2
    multi_stub_filter.Ports.coordinates[1][0] = pol_list[0].exterior.coords[1][0] * 1e-3
    multi_stub_filter.Ports.coordinates[1][1] = pol_list[0].exterior.coords[3][1] * 1e-3 / 2

    if debug is True:
        multi_stub_filter.layout_plot()

    # write modified design in the new_proj folder
    multi_stub_filter.write_new()

    # simulate modified design from new_proj folder
    multi_stub_filter.simulate_new()

    # READ FREQUENCY RESPONSE
    freq, s21 = multi_stub_filter.get_s21()

    if debug is True:
        plt.plot(freq, filtering.get_magnitude(s21))
        plt.show()

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
    # the first polygon of the list is the main transmission line
    p1 = [(0, 0), (line_length, 0), (line_length, line_width), (0, line_width)]
    polygons.append(Polygon(p1))

    x0 = pre_transmission_line
    stub_distance.append(0)
    for i in range(num_stubs):
        if stub_length[i] > 0:
            pol_shell = [(x0, line_width), (x0 + stub_width[i], line_width),
                       (x0 + stub_width[i], line_width + stub_length[i]), (x0, line_width + stub_length[i])]
        else:
            pol_shell = [(x0, 0), (x0 + stub_width[i], 0),
                       (x0 + stub_width[i], stub_length[i]), (x0, stub_length[i])]
        x0 = x0 + stub_width[i] + stub_distance[i]
        polygons.append(Polygon(pol_shell))
    return polygons


def mask_distance(response, interval, value):    # , bound='high'):
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


l = np.random.uniform(-6, 6, 7).tolist()
w = np.random.uniform(0.2, 1, 7).tolist()
d = np.repeat(3, 6).tolist()
layout_polygons = build_geometry(l, w, d)


# RUN PARAMETRIZED EXAMPLE
f, s = run_parametric(layout_polygons, debug=True)

# surrounding box area
merged_layout = cascaded_union(layout_polygons)
ar = box(*list(merged_layout.bounds)).area
print(ar)

