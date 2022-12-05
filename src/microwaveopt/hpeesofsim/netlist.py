import os
import sys
from copy import copy


class Lib(object):
    def __init__(self, line):
        line = line.split()
        self.command = line[0].strip('\"')
        self.name = line[1].strip('\"')
        self.comp = line[3].strip('\"')


class Component(object):
    def __init__(self):
        self.name = []
        self.label = []
        self.pins = []
        self.properties = {}
        self.lib = None

    def add(self, name, label, pins, properties=None, lib=None):
        pass


class Netlist(object):
    def __init__(self):
        self.net_file = "netlist.log"
        self.file_lines = []
        self.header = []
        self.components_list = []
        self.components = {}
        self.libs = []

    @staticmethod
    def load(folder):
        net = Netlist()
        path = os.path.join(folder, net.net_file)
        try:
            net_file = open(path, 'r')
        except OSError:
            print("Could not open/read Netlist file:", path)
            sys.exit()

        with net_file:
            lines = net_file.readlines()
            net.file_lines = lines
            # Concatenate new-lines
            linit = 0
            while linit < len(lines):
                if lines[linit] == '\n':
                    lines.pop(linit)
                if len(lines[linit]) > 1 and lines[linit][-2] == '\\':
                    lines[linit] = lines[linit][:-1]
                    while lines[linit+1][-2] == '\\':
                        lines[linit] = lines[linit][:-1] + lines[linit + 1][:-1]
                        lines.pop(linit + 1)
                    lines[linit] = lines[linit][:-1] + lines[linit + 1][:-1]
                    lines.pop(linit + 1)
                linit += 1

            for li, ln in enumerate(lines):
                split_line = ln.strip().split()
                if split_line[0] == ";":
                    net.header.append(ln)
                elif ln[0] == '#':
                    lib = Lib(ln)
                    net.libs.append(lib)
                else:
                    comp = Component()

                    comp.name = split_line[0]
                    j = 1
                    if ':' in split_line[0] or split_line[0] == 'model':
                        if split_line[0] != 'model':
                            comp.name, comp.label = comp.name.split(':')
                        else:
                            comp.label = 'model'
                        while '=' not in split_line[j]:
                            comp.pins.append(split_line[j].strip())
                            j = j + 1
                            if j > len(split_line):
                                break
                    else:
                        comp.label = copy(comp.name)

                    prop_strings = split_line[j:]
                    for pi, p in enumerate(prop_strings):
                        if '=' not in p:
                            prop_strings[pi - 1] = prop_strings[pi - 1] + ' ' + p
                            prop_strings.pop(pi)
                    prop_keys = [s.split(sep='=')[0] for s in prop_strings]
                    prop_values = [s.split(sep='=')[1] for s in prop_strings]
                    comp.properties = dict(zip(prop_keys, prop_values))

                    net.components_list.append(comp)

            for c in net.components_list:
                list_lib = [lib.comp for lib in net.libs]
                if c.name in list_lib:
                    c.lib = net.libs[list_lib.index(c.name)]

            net.components = dict(zip([c.label for c in net.components_list], net.components_list))
            return net

    def write(self, folder, write_lines=False):
        path = os.path.join(folder, self.net_file)
        try:
            net_file = open(path, 'w')
        except OSError:
            print("Could not open/read Netlist file:", path)
            sys.exit()

        with net_file:
            lines = self.file_lines
            if write_lines:
                net_file.writelines(lines)

            else:
                net_file.writelines(self.header)
                lines = []
                for c in self.components_list:
                    if c.lib:
                        lib_line = c.lib.command + ' ' + f'\"{c.lib.name}\"' + ' , ' + f'\"{c.lib.comp}\"'
                        lines.append(lib_line + '\n')

                    if c.name != c.label:
                        line = c.name + ':' + c.label
                    else:
                        line = c.name

                    if c.pins:
                        for p in c.pins:
                            line = line + ' ' + p

                    for k,v in zip(c.properties.keys(), c.properties.values()):
                        line = line + ' ' + k + '=' + v

                    lines.append(line + '\n')

                net_file.writelines(lines)
