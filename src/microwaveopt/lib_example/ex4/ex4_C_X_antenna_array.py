import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity

import microwaveopt.em_lib.filtering as filtering
from microwaveopt.momentum.design import Design
from microwaveopt.momentum.em_setup import Sampling


def set_dim(polygon, value, dim=0, fix=None):
    """
    Function to update one dimension of a polygon, adding "value" to the coordinates. Runs on every Polygon, but it's
    specific for rectangles that are normal to the x/y axis.
    ARGUMENTS:
        . polygon: rectangle to be modified
        . value: amount to be added to the dimension
        . dim: dimension to be modified (dim=0 --> x, dim=1 --> y)
        . fix: set the direction of polygon expansion:
            fix = None --> expansion of both side of the axis
            fix = 0 --> expansion toward +infinite
            fix = 1 --> expansion toward -infinite
    """
    points = [list(i) for i in list(polygon.exterior.coords)]
    for i in range(len(points)):
        if fix == 0:
            if points[i][dim] > polygon.centroid.coords[0][dim]:
                points[i][dim] += value
        elif fix == 1:
            if points[i][dim] < polygon.centroid.coords[0][dim]:
                points[i][dim] -= value
        else:
            if points[i][dim] > polygon.centroid.coords[0][dim]:
                points[i][dim] += value / 2
            else:
                points[i][dim] -= value / 2
    points = [(i[0], i[1]) for i in points]
    return Polygon(points)


def set_width(polygon, value, fix=None):
    fix_dic = {"left": 0, "right": 1}
    return set_dim(polygon, value[0], 0, fix_dic.get(fix))


def set_height(polygon, value, fix=None):
    fix_dic = {"bottom": 0, "top": 1}
    return set_dim(polygon, value[1], 1, fix_dic.get(fix))


def strecth_x(geom, i, params, fix=None, drag=[]):
    geom[i] = set_width(geom[i], params[i], fix=fix)
    off = params[i][0]
    if fix == 'left':
        off = params[i][0]
    elif fix == 'right':
        off = -params[i][0]
    for p in drag:
        geom[p] = affinity.translate(geom[p], xoff=off)


def strecth_y(geom, i, params, fix=None, drag=[]):
    geom[i] = set_height(geom[i], params[i], fix=fix)
    off = params[i][1]
    if fix == 'bottom':
        off = params[i][1]
    elif fix == 'top':
        off = -params[i][1]
    for p in drag:
        geom[p] = affinity.translate(geom[p], yoff=off)


def set_params(old_geometry, params):
    """
    Function to add a term to the width and height of each polygon of the geometry. Runs on every Polygon, but it's
    specific for rectangles that are normal to the axis.
    ARGUMENTS:
        old_geometry: list of polygon from the initial Design class
        params: [[w0,h0],... ,[wn,hn]]: list of pairs, amounts to be added to width and height of each polygon
                                        fill one pair with zeros to keep the corresponding Polygon unchanged
    Note that some Polygons are not currenty updated, so some values in params are not used. They can be added, removed
    or re-numbered to use a new design or layout
    CURRENTLY USED POLYGONS:
    [0, 2, 4, 6, 7, 9, 11, 12, 14, 16, 20] for a maximum of 22 available parameters
    """

    assert len(old_geometry) == len(params), "ERROR! different number of polygon and specified parameters"
    for i in params:
        assert len(i) == 2, f"ERROR! Number of specified dimensions is different from 2 for Polygon {i}"

    new = [Polygon(p.shapely.exterior.coords) for p in old_geometry]

    new[0] = set_width(new[0], params[0])
    strecth_y(new, 0, params, fix='bottom', drag=[1])

    new[2] = set_width(new[2], params[2])
    strecth_y(new, 2, params, fix='bottom', drag=[5])

    new[12] = set_width(new[12], params[12])
    strecth_y(new, 12, params, fix='bottom', drag=[13])

    new[16] = set_width(new[16], params[16])
    strecth_y(new, 16, params, fix='bottom', drag=[17])

    new[4] = set_height(new[4], params[4])
    strecth_x(new, 4, params, fix='left', drag=[0,1,18])

    new[6] = set_height(new[6], params[6])
    strecth_x(new, 6, params, fix='left', drag=[3,2,5,4,18,0,1])

    new[9] = set_height(new[9], params[9])
    strecth_x(new, 9, params, fix='left', drag=[6,3,2,5,4,18,0,1])

    new[14] = set_height(new[14], params[14])
    strecth_x(new, 14, params, fix='right', drag=[15,16,17])

    new[20] = set_height(new[20], params[20])
    strecth_x(new, 20, params, fix='right', drag=[12,13,14,15,16,17,19])

    new[11] = set_height(new[11], params[11])
    strecth_x(new, 11, params, fix='right', drag=[12,13,14,15,16,17,19,20])

    new[7] = set_width(new[7], params[7])
    strecth_y(new, 7, params, fix='bottom', drag=[8])

    return new


ex_path = __file__
ex_dir = os.path.abspath(os.path.join(ex_path, os.pardir))
init_proj = os.path.join(ex_dir, "ADS_Original_Files")
new_proj = os.path.join(ex_dir, "ADS_Parametric_Layout")

antenna_array = Design(init_proj, new_proj)
antenna_array.load_original()
antenna_array.layout_plot(label=True) #, coords=True)

# antenna_array.layout_plot(label=True, coords=False)

# Modifiying layout
old_geom = antenna_array.Layout.polygons
# set variation amount for each parametrized polygon
dims = [[0, 0] for i in range(len(old_geom))]
dims[0][0] = 3
dims[0][1] = 10
dims[2][0] = 3
dims[2][1] = 20
dims[4][0] = 0
dims[4][1] = 1
dims[6][0] = 10
dims[6][1] = -1
dims[7][0] = 5
dims[7][1] = 20
dims[9][0] = 10
dims[9][1] = 10
dims[11][0] = 10
dims[11][1] = 10
dims[12][0] = 10
dims[12][1] = 10
dims[14][0] = 20
dims[14][1] = 0
dims[16][0] = 10
dims[16][1] = 10
dims[20][0] = 10
dims[20][1] = 10

new_geom = set_params(old_geom, dims)
antenna_array.Layout.overwrite_geometry(new_geom, mask='P17')

# Updates port with the new position on the feed Polygon
antenna_array.Ports.coordinates[0][0] = (new_geom[8].exterior.coords[0][0] + new_geom[8].exterior.coords[1][0])*1e-3 / 2
antenna_array.Ports.coordinates[0][1] = new_geom[8].exterior.coords[3][1] * 1e-3


antenna_array.layout_plot(label=True) #, coords=True)

antenna_array.write_new()
antenna_array.simulate_new(quiet=False)

freq, s11 = antenna_array.get_s11()
plt.plot(freq, filtering.get_magnitude(s11))
plt.show()


















