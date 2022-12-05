import os
from microwaveopt.utils import find_line, find_arg
import warnings

# RESULTS DATA CLASS #################################################################################################
#######################################################################################################################
#######################################################################################################################
class Data(object):
    def __init__(self, name, unit, value_number):
        self.name = name
        self.unit = unit
        self.value_number = value_number
        self.values = []


# SIMULATION RESULTS CLASS ############################################################################################
#######################################################################################################################
#######################################################################################################################
class EMResults(object):
    def __init__(self):
        self.file_linear = 'proj.cti'
        self.file_adaptive = 'proj.afs'
        self.citifile = []
        self.name = []
        self.variables = []
        self.data = []

    @staticmethod
    def load(folder, afs=False):
        res = EMResults()
        if afs:
            path = os.path.join(folder, res.file_adaptive)
        else:
            path = os.path.join(folder, res.file_linear)

        if os.path.exists(path):
            l_file = open(path, 'r')

            lines = l_file.readlines()

            ln = find_line(lines, 'CITIFILE')[0]
            res.citifile = find_arg(lines[ln], 'CITIFILE', separator=' ')

            ln = find_line(lines, 'NAME')[0]
            res.name = find_arg(lines[ln], 'NAME', separator=' ')

            l_var = find_line(lines, "VAR ")
            first = find_line(lines, "VAR_LIST_BEGIN")
            last = find_line(lines, "VAR_LIST_END")

            if not (len(l_var) == len(first) == len(last)):
                raise ValueError("ERROR! Variables list is not correctly formatted in file proj.cti")

            for i in range(len(l_var)):
                args = lines[l_var[i]].split()
                values = lines[first[i] + 1:last[i]]
                values = [float(v.strip()) for v in values]
                var = Data(args[1], args[2], args[3])
                var.values = values
                res.variables.append(var)

            l_data = find_line(lines, "DATA")
            first = find_line(lines, "BEGIN")
            last = find_line(lines, "END")
            if not (len(l_data) == len(first) == len(last)):
                raise ValueError("ERROR! Variables list is not correctly formatted in file proj.cti")
            for i in range(len(l_data)):
                args = lines[l_data[i]].split()
                values = lines[first[i] + 1:last[i]]
                value_number = last[i] - first[i] - 1
                if args[2] == 'RI':
                    tmp = []
                    for v in values:
                        a = v.split(',')
                        re = float(a[0])
                        im = float(a[1])
                        tmp.append(complex(re, im))

                data = Data(args[1], args[2], value_number)
                data.values = tmp
                res.data.append(data)
            return res
        else:
            warnings.warn(f"Simulation results file not available: {path}")
# res = EMResults.load('ex', sampling='adaptive')
# print()
#
