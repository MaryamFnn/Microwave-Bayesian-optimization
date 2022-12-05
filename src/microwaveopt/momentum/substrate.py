"""
CLASSES FOR ADS SUBTRATE DEFINITION
"""
import sys
import os
from microwaveopt.utils import find_line, find_arg
import copy

# SUPPORT CLASSES #####################################################################################################
#######################################################################################################################
#######################################################################################################################
class Unit(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    @staticmethod
    def read(line):
        line = line.split('=')
        name = line[0].strip()
        value = line[1].strip()
        un = Unit(name, value)
        return un

    def print(self):
        line = "  " + self.name.upper() + "=" + self.value.upper()
        return line + '\n'


class Material(object):
    """
    do nor
    """
    def __init__(self, line=None):
        if line is not None:
            self.name = find_arg(line, 'MATERIAL', separator=' ', terminator=' ')
            self.permittivity = find_arg(line, 'PERMITTIVITY', separator='=', terminator=' ')
            self.losstangent = find_arg(line, 'LOSSTANGENT', separator='=', terminator=' ')
            self.permeability = find_arg(line, 'PERMEABILITY', separator='=', terminator=' ')
            self.imag_permeability = find_arg(line, 'IMAG_PERMEABILITY', separator='=', terminator=' ')
            self.conductivity = find_arg(line, 'CONDUCTIVITY', separator='=', terminator=' ')

            if 'DJORDJEVIC' in line:
                self.djordjevic = True
                self.lowfreq = find_arg(line, 'LOWFREQ', separator='=', terminator=' ')
                self.valuefreq = find_arg(line, 'VALUEFREQ', separator='=', terminator=' ')
                self.highfreq = find_arg(line, 'HIGHFREQ', separator='=', terminator=' ')

    def print(self):
        attributes = copy.copy(self.__dict__)
        line = f"  MATERIAL {self.name}"
        attributes.pop('name')

        if 'djordjevic' in attributes:
            attributes.pop('djordjevic')
            attributes.pop('lowfreq')
            attributes.pop('valuefreq')
            attributes.pop('highfreq')

        for att in attributes:
            if attributes[att] is not None:
                line += f" {att.upper()}={attributes[att]}"

        if 'djordjevic' in self.__dict__:
            line += f" DJORDJEVIC LOWFREQ={self.lowfreq} VALUEFREQ={self.valuefreq} HIGHFREQ={self.highfreq}"

        return line + '\n'


class Mask(object):
    def __init__(self, name, number, material=None, precedence=None, color=None, operation=None, mask_prop=None, intrude=None):
        self.number = number
        self.name = name
        self.precedence = precedence
        if color is None:
            self.color = 'ffff00'
        else:
            self.color = color
        # if material is None:
        #     self.material = 'PERFECT CONDUCTOR'
        if operation is not None:
            self.material = material
        if operation is not None:
            self.operation = operation
        if mask_prop is not None:
            self.mask_properties = mask_prop
        if intrude is not None:
            self.intrude = 'INTRUDE=1e-06 UP'

    @staticmethod
    def read(line):
        number = find_arg(line, 'MASK', separator=' ', terminator=' ')
        name = find_arg(line, 'name', separator='=', terminator=' ')

        precedence = find_arg(line, 'PRECEDENCE', separator='=', terminator=' ')
        negative_prec = find_arg(line, 'NEGATIVE PRECEDENCE', separator='=', terminator=' ')
        if negative_prec is not None:
            precedence = 'NEGATIVE PRECEDENCE=' + negative_prec.strip()
        if (precedence is not None) and (negative_prec is None):
            precedence = 'PRECEDENCE=' + precedence.strip()

        operation = find_arg(line, 'OPERATION', separator='=', terminator=' ')

        color = find_arg(line, 'COLOR', separator='=', terminator=' ')
        material = find_arg(line, 'MATERIAL', separator='=', terminator=' ')
        mask_prop = find_arg(line, 'MASK_PROPERTIES', separator='=', terminator=' ')
        intrude = find_arg(line, 'INTRUDE', separator='=',  terminator=' ')
        if intrude is not None:
            up = find_arg(line, 'UP', separator='',  terminator=' ')
            if up is not None:
                intrude = intrude + " UP"
            else:
                intrude = intrude + " DOWN"
        mk = Mask(name,
                  number,
                  material=material,
                  precedence=precedence,
                  color=color,
                  operation=operation,
                  mask_prop=mask_prop,
                  intrude=intrude)
        return mk

    def print(self):
        attributes = self.__dict__
        line = f" MASK {self.number}"
        for att in attributes:
            if (attributes[att] is not None) and (att!='number') and (att!='precedence'):
                line += f" {att.upper()}={attributes[att]}"
        if self.precedence is not None:
            line += " " + self.precedence
        return line + '\n'


class Layer(object):
    def __init__(self, l_type= None, name=None, mask=None, height=None, material=None):
        self.l_type = l_type
        if l_type is None:
            raise ValueError("ERROR! Layer type not correctly specified")
        self.name = name
        self.mask = mask
        self.name = name
        self.height = height
        self.material = material


    @staticmethod
    def read(line):
        first_word = line.strip().split()
        if (first_word[0] == "TOP") or (first_word[0] == "BOTTOM"):
            l_type = first_word[0] + ' ' + first_word[1]
        else:
            l_type = first_word[0]
        name = find_arg(line, 'name', separator='=', terminator=' ')
        mask = find_arg(line, 'mask', separator='=', terminator=' ')
        height = find_arg(line, 'height', separator='=', terminator=' ')
        material = find_arg(line, 'material', separator='=', terminator=' ')
        layer_obj = Layer(l_type=l_type, name=name, mask=mask, height=height, material=material)
        return layer_obj

    def print(self):
        attributes = self.__dict__
        line = f"  {self.l_type}"
        for att in attributes:
            if (attributes[att] is not None) and (att != "l_type"):
                line += f" {att.upper()}={attributes[att]}"
        return line + '\n'


# MAIN CLASS ##########################################################################################################
#######################################################################################################################
#######################################################################################################################
class Substrate(object):
    def __init__(self):
        self.techformat = 'V2'
        self.units = []
        self.materials = []
        self.operations = []
        self.masks = []
        self.stack = []
        self.sub_file = 'proj.ltd'

    def add_material(self, line):
        mt = Material(line)
        self.materials.append(mt)
        return self.materials

    def add_mask(self, name, mk_num=None, material='PERFECT CONDUCTOR'):
        if mk_num is None:
            mk_num = len(self.masks) + 1
        mk = Mask(name, mk_num, material)
        self.masks.append(mk)
        return self.masks

    def add_layer(self, l_type, name, position, mask=None, height=None, material = None):
        layer = Layer(l_type=l_type, name=name, mask=mask, height=height, material=material)
        self.stack.insert(position, layer)
        return self.stack

    @staticmethod
    def load(folder):
        sub = Substrate()
        path = os.path.join(folder, sub.sub_file)
        try:
            sub_file = open(path, 'r')
        except OSError:
            print("Could not open/read Substrate file:", path)
            sys.exit()

        with sub_file:
            lines = sub_file.readlines()

            sub.techformat = find_arg(lines[0], 'TECHFORMAT', separator='=', terminator='\n')

            first = find_line(lines, 'UNITS')[0]
            last = find_line(lines, 'END_UNITS')[0]
            for l in lines[first + 1:last]:
                sub.units.append(Unit.read(l))

            first = find_line(lines, 'BEGIN_MATERIAL')[0]
            last = find_line(lines, 'END_MATERIAL')[0]
            for l in lines[first + 1:last]:
                sub.materials.append(Material(l))

            first = find_line(lines, 'BEGIN_OPERATION')[0]
            last = find_line(lines, 'END_OPERATION')[0]
            for l in lines[first + 1:last]:
                sub.operations.append(l)

            first = find_line(lines, 'BEGIN_MASK')[0]
            last = find_line(lines, 'END_MASK')[0]
            for l in lines[first + 1:last]:
                sub.masks.append(Mask.read(l))

            first = find_line(lines, 'BEGIN_STACK')[0]
            last = find_line(lines, 'END_STACK')[0]
            for l in lines[first + 1:last]:
                sub.stack.append(Layer.read(l))

            return sub


    def write(self, folder):
        path = os.path.join(folder, 'proj.ltd')
        try:
            sub_file = open(path, 'w')
        except OSError:
            print("Could not open/read Substrate file:", path)
            sys.exit()

        with sub_file:
            lines = []
            lines.append(f"TECHFORMAT={self.techformat}\n")
            lines.append('\n')
            
            lines.append("UNITS\n")
            for u in self.units:
                lines.append(u.print())
            lines.append("END_UNITS\n\n")

            lines.append("BEGIN_MATERIAL\n")
            for m in self.materials:
                lines.append(m.print())
            lines.append("END_MATERIAL\n\n")

            lines.append("BEGIN_OPERATION\n")
            for op in self.operations:
                lines.append(op)
            lines.append("END_OPERATION\n\n")

            lines.append("BEGIN_MASK\n")
            for m in self.masks:
                lines.append(m.print())
            lines.append("END_MASK\n\n")

            lines.append("BEGIN_STACK\n")
            for ls in self.stack:
                lines.append(ls.print())
            lines.append("END_STACK\n\n")

            sub_file.writelines(lines)






#
# sub1.add_material('sio2', 3)
# sub1.add_mask('cond1')
# sub1.add_layer('diel', 1, 5e-4, material='sio2')
# sub1.add_interface('int1', 2, mask='cond1')
# sub1.add_mask('cond2')
#
# sub1= Substrate.load('ex')
# sub1.write('ex')
