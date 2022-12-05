import os
from microwaveopt.utils import find_line, find_arg
import warnings

class Variable(object):
    def __init__(self):
        self.number = []
        self.name = []
        self.quantity = []
        self.type = []
        self.indep = []
        self.mixop = None
        self.data = []

    def read(self, var_line):
        self.type = find_arg(var_line, 'type', separator='=')
        self.indep = find_arg(var_line, 'indep', separator='=')
        self.mixop = find_arg(var_line, 'mixop', separator='=')
        var_line = var_line.split()
        if 'Variables:' in var_line:
            var_line.pop(0)
        self.number = var_line[0]
        self.name = var_line[1]
        self.quantity = var_line[2]
        return self

# SIMULATION RESULTS CLASS ############################################################################################
#######################################################################################################################
#######################################################################################################################
class Spectra(object):
    def __init__(self):
        self.file_raw = 'spectra.raw'
        self.citifile = []
        self.name = []
        self.variables_list = []
        self.variables = {}
        self.data = []

    # def __getattr__(self, attr):
    #     return self.variables[attr].data

    @staticmethod
    def load(folder):
        res = Spectra()
        path = os.path.join(folder, res.file_raw)

        if os.path.exists(path):
            l_file = open(path, 'r')

            lines = l_file.readlines()

            var_init = find_line(lines, 'Variables:')[0]
            var_stop = find_line(lines, 'Values:')[0]
            file_stop = find_line(lines, '#')[0]

            for i in range(var_init, var_stop):
                var_line = lines[i].strip().split()
                v = Variable()
                v.read(lines[i].strip())
                res.variables_list.append(v)

            full_sequence = ''
            for i in range(var_stop, file_stop):
                full_sequence = full_sequence + lines[i][:-1]
            full_sequence = full_sequence.split()[1:]

            for vi, v in enumerate(res.variables_list):
                idx = [i for i in range(vi, len(full_sequence), 6)]
                data_curr = [float(full_sequence[i]) for i in idx]
                v.data = data_curr

            res.variables = dict(zip([v.name for v in res.variables_list], res.variables_list))

            return res
        else:
            raise ValueError(f"Simulation results file not available: {path}")

