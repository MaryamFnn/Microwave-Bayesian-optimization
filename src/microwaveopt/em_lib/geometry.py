'''
COLLECTION OF FUNCTIONS TO MODIFY SHAPELY-BASED LAYOUT
'''
from shapely.geometry import Polygon
from shapely import affinity


def set_dim(polygon, value, dim=0, fix=None):
    """
    Function to update one dimension of a polygon, adding "value" to the coordinates. Runs on every Polygon, but it's
    specific for rectangles that are normal to the axis.
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