import os
import sys
from microwaveopt.utils import find_line, find_arg
from shapely.geometry import Polygon

# class Lpoly(object):
#     def __init__(self, number, mask, width, x_points, y_points):
#         self.mask = mask
#         self.width = width
#         self.x_points = x_points
#         self.y_points = y_points
#         self.number = number
#         self.association = None
#         self.ass_net = None


class Lpoly_shape(object):
    def __init__(self, shell, number, mask, width):
        self.shapely = Polygon(shell)
        self.mask = mask
        self.width = width
        self.number = number
        self.association = None
        self.ass_net = None
        self.x_points = []
        self.y_points = []
        for p in shell:
            self.x_points.append(p[0])
            self.y_points.append(p[1])


class Layout(object):
    """
    USAGE:
    pa = ProjA()            : class
    pa.load()               : read all from file
    pa.units                : return ads unit scale
    pa.edit                 : return edit name
    pa.polygons[i].name     : return name of polygon i
    pa.polygons[i].width    : return width of polygon i
    pa.polygons[i].points   : return points coordinates of polygon i

    """

    def __init__(self):
        self.layout_file = "proj_a"
        self.units = []
        self.edit = []
        self.polygons = []

    def shapely(self, mask=None):
        if mask is not None:
            return [p.shapely for p in self.polygons if p.mask == mask]
        else:
            return [p.shapely for p in self.polygons]

    @staticmethod
    def load(folder):
        ly = Layout()
        path = os.path.join(folder, ly.layout_file)
        try:
            l_file = open(path, 'r')
        except OSError:
            print("Could not open/read file:", path)
            sys.exit()
        with l_file:
            lines = l_file.readlines()

            ln = find_line(lines, 'UNITS')[0]
            ly.units = find_arg(lines[ln], 'UNITS', separator=' ', terminator=';')

            ln = find_line(lines, 'EDIT')[0]
            ly.edit = find_arg(lines[ln], 'EDIT', separator=' ', terminator=';')

            l_add_p = find_line(lines, 'ADD P')
            l_add_n = find_line(lines, 'ADD N')

            for i in range(len(l_add_p)):
                mask = find_arg(lines[l_add_p[i]], 'ADD', separator=' ', terminator=' ')
                width = find_arg(lines[l_add_p[i]], 'W', separator='', terminator=' ')
                all_points = find_arg(lines[l_add_p[i]], 'W' + width, separator='  ', terminator=';')
                all_points = all_points.split(' ')
                # x_points = []
                # y_points = []
                # for p in all_points:
                #     x_points.append(float(p.split(',')[0]))
                #     y_points.append(float(p.split(',')[1]))

                # curr_pol = Lpoly(i, mask, float(width), x_points, y_points)

                polygon_shell = []
                for p in all_points:
                    xp = float(p.split(',')[0])
                    yp = float(p.split(',')[1])
                    polygon_shell.append((xp, yp))
                curr_pol = Lpoly_shape(polygon_shell, i, mask, float(width))

                if (l_add_p[i] + 2) in l_add_n:
                    curr_pol.association = lines[l_add_p[i] + 2]
                    curr_pol.ass_net = find_arg(curr_pol.association, 'net', separator='=', terminator='\'')
                ly.polygons.append(curr_pol)
            return ly

    def write(self, folder):
        new_path = os.path.join(folder, self.layout_file)
        try:
            w_file = open(new_path, 'w+')
        except OSError:
            print("Could not open/read file:", new_path)
            sys.exit()
        with w_file:
            w_file.write('UNITS %s;' % self.units)
            w_file.write('\rEDIT %s;\n' % self.edit)

            for pol in self.polygons:
                w_file.write('ADD %s :W%4.6f ' % (pol.mask, pol.width))
                for px, py in zip(pol.x_points, pol.y_points):
                    w_file.write(' %6.4f,%6.4f' % (px, py))

                # # Alternative Write from shapely coordinates
                # for coords in pol.exterior.coords[:-1]:
                #     w_file.write(' %6.4f,%6.4f' % (coords[0], coords[1]))

                w_file.write(";\n")
                if pol.association is not None:
                    w_file.write("  BEGIN_ASSOC\n")
                    w_file.write(f"{pol.association}")
                    w_file.write("  END_ASSOC\n")

            w_file.write('SAVE;\r\n')

    def overwrite_geometry(self, pol_list, mask=None):
        width = 0
        if mask is None:
            raise ValueError("ERROR! Layer mask to overwrite is not specified")
        j = -1
        for i in range(len(self.polygons)):
            if self.polygons[j].mask == mask:
                width = self.polygons[j].width
                self.polygons.pop(j)
            else:
                j -= 1
        for idx, pol in enumerate(pol_list):
            shell = pol.exterior.coords[:-1]
            # x_coord = [point[0] for point in pol]
            # y_coord = [point[1] for point in pol]
            curr_pol = Lpoly_shape(shell, idx, mask, width)
            self.polygons.append(curr_pol)
        return self






