import sys
import os
import xml.etree.ElementTree as ET
from microwaveopt.utils import find_arg, find_line
import cmath
import numpy as np

# SIMULATION PORTS CLASS ##############################################################################################
#######################################################################################################################
#######################################################################################################################
class Ports(object):
    """
    USAGE:
    - use .load(folder) to create Ports object, automatically reads from /folder/proj.pin;
    - modify ports values in self.coordintates;
    - use .write(folder) method to save on new port file
    """
    def __init__(self):
        self.ports_file = 'proj.pin'
        self.coordinates = []
        self.txt = []
        self.tree = None
        self.number = 0

    def update_tree(self):
        xml_root = self.tree.getroot()
        x_ports = xml_root.findall("./pin/layout/shape/point/x")
        y_ports = xml_root.findall("./pin/layout/shape/point/y")

        for i in range(self.number):
            x_ports[i].text = f"{ self.coordinates[i][0]}"
            y_ports[i].text = f"{ self.coordinates[i][1]}"


    @staticmethod
    def load(folder):
        ps = Ports()
        path = os.path.join(folder, ps.ports_file)
        try:
            ps.tree = ET.parse(path)
        except OSError:
            print("Could not open/read file:", path)
            sys.exit()

        xml_root = ps.tree.getroot()
        x_ports = xml_root.findall("./pin/layout/shape/point/x")
        y_ports = xml_root.findall("./pin/layout/shape/point/y")

        # if (len(x_ports) != self.port_number) | (len(y_ports) != self.port_number):
        #     raise ValueError( "*** ERROR! Number of ports is different from the one specified  HALTING. ***\n")

        ps.number = len(x_ports)
        for i in range(ps.number):
            x_val = float(x_ports[i].text)
            y_val = float(y_ports[i].text)
            ps.coordinates.append([x_val, y_val])

        return ps

    def write(self, folder):
        self.update_tree()

        new_path = os.path.join(folder, self.ports_file)
        try:
            # my_file = open(self.path, 'r')
            self.tree.write(new_path, encoding='utf-8', xml_declaration=True)
        except OSError:
            print(f"Could not write on file {new_path}")
            sys.exit()

        return new_path

    def move(self, port_number, position):
        if not isinstance(position, list):
            raise ValueError("Error: the new give position is not a pair of coordinates \n")
        self.coordinates[port_number-1][0] = position[0]
        self.coordinates[port_number-1][1] = position[1]
        self.update_tree()
        return self.coordinates


# FREQUENCY SAMPLING CLASS ############################################################################################
#######################################################################################################################
#######################################################################################################################
class Sampling(object):
    def __init__(self, mode=None, lower=None, higher=None, step=None, max_samples=None):
        self.sim_file = 'proj.sti'
        self.mode = mode
        self.lower = lower
        self.higher = higher
        self.step = step
        self.max_samples = max_samples
        if mode == 'linear' and (step is None):
            raise ValueError("ERROR! Step size is non defined in linear sampling")
        if mode == 'adaptive' and (max_samples is None):
            raise ValueError("ERROR! Max number of samples is non defined in adaptive sampling")




    @staticmethod
    def load(folder):
        samp = Sampling()
        path = os.path.join(folder, samp.sim_file)
        try:
            sub_file = open(path, 'r')
        except OSError:
            print("Could not open/read Substrate file:", path)
            sys.exit()

        with sub_file:
            lines = sub_file.readlines()
            afs_line = find_line(lines, "AFS")
            if afs_line:
                mode = 'adaptive'
                lower = float(find_arg(lines[afs_line[0]-1], "START", separator=' ', terminator='STOP'))
                higher = float(find_arg(lines[afs_line[0]-1], "STOP", separator=' ', terminator='STEP'))
                max_samples = float(find_arg(lines[afs_line[0]], "MAXSAMPLES", separator=' ', terminator=' ').strip())
                samp = Sampling(mode=mode, lower=lower, higher=higher, max_samples=max_samples)
            else:
                mode = 'linear'
                lower = float(find_arg(lines[0], "START", separator=' ', terminator=' ').strip())
                higher = float(find_arg(lines[0], "STOP", separator=' ', terminator=' ').strip())
                step = float(find_arg(lines[0], "STEP", separator=' ', terminator=',').strip())
                samp = Sampling(mode=mode, lower=lower, higher=higher, step=step)
            return samp

    def write(self, folder):
        new_path = os.path.join(folder, self.sim_file)
        try:
            if os.path.exists(new_path):
                os.remove(new_path)
            w_file = open(new_path, 'w+')
        except OSError:
            print(f"Could not write on file {new_path}")
            sys.exit()
        lines = []
        assert self.lower is not None, "ERROR! START frequency not set"
        assert self.lower is not None, "ERROR! STOP frequency not set"

        if self.mode == 'linear':
            lines.append(f"START {self.lower} STOP {self.higher} STEP {self.step}, \n")
        elif self.mode == 'adaptive':
            lines.append(f"START {self.lower} STOP {self.higher} STEP 1, \n")
            lines.append(f"AFS S_50 MAXSAMPLES {self.max_samples} SAMPLING ALL NORMAL;")
        else:
            raise ValueError("Sampling mode not recognized")

        w_file.writelines(lines)
        w_file.close()
        return new_path


# FAR FIELD CLASS ############################################################################################
#######################################################################################################################
#######################################################################################################################

class FarFields(object):
    def __init__(self, folder, freq_step, ports, voltage, impedance, visualization=1, file_name='proj.fff'):
        self.vpl_file = 'proj.vpl'
        self.ff_file = file_name
        self.freq_step = freq_step
        file_path = os.path.join(folder, self.vpl_file)

        if os.path.isfile(file_path):
            os.remove(file_path)
        try:
            vpl = open(file_path, 'w')
        except OSError:
            print("Could not create FarFields plan file:", file_path)
            sys.exit()

        with vpl:
            script = [f"VISUALIZATIONTYPE {visualization}; # 3D output\n",
                   f"PARAMETER FREQUENCY, UNITS GHz,\n",
                   f"PT {freq_step};\n",
                   f"PARAMETER PHI, UNITS DEG, PT 0; # Needed, but not used for 3D output\n",
                   f"VAR THETA, UNITS DEG, PT 0; # Needed, but not used for 3D output\n"]

            for p, v, z in zip(ports, voltage, impedance):
                script2 = [f"PORT {p},\n",
                        f"UNITS VOLT, UNITS DEG, AMPLITUDE {abs(v)}, PHASE {cmath.phase(v)},\n",
                        f"UNITS OHM, UNITS RAD, AMPLITUDE {abs(z)}, PHASE {cmath.phase(z)};\n"]
                script = script + script2

            vpl.writelines(script)
            vpl.close()

    def load(self, folder):
        file_path = os.path.join(folder, self.ff_file)
        try:
            ff_file = open(file_path, 'r')
        except OSError:
            print("Could not create FarFields plan file:", file_path)
            sys.exit()

        with ff_file:
            lines = ff_file.readlines()
            data_lines = []
            for l in lines:
                if (l[0] != '#') and (l[0] != '\n'):
                    l = l.strip().split('  ')
                    l = [float(num) for num in l ]
                    data_lines.append(l)
            data_lines = np.asarray(data_lines)

        return data_lines







