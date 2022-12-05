import os
from distutils.dir_util import copy_tree

from microwaveopt.momentum.substrate import Substrate
from microwaveopt.momentum.em_setup import Ports, Sampling, FarFields
from microwaveopt.momentum.layout import Layout, Lpoly_shape
from microwaveopt.momentum.sim_res import EMResults
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import subprocess
from ..config import HPEESOF_DIR, ADS_LICENSE_FILE

class Design(object):
    """
    ADS Momentum design class
    """

    def __init__(self, folder1, folder2, software='momentum'):
        assert os.path.isdir(folder1), "ERROR! Initial project folder does not exist"
        assert os.path.isdir(folder2), "ERROR! New project folder does not exist"
        self.init_proj_dir = folder1
        self.new_proj_dir = folder2
        self.ads_path = None
        self.ads_license = None

        self.Substrate = None
        self.Layout = None
        self.Ports = None
        self.Sampling = None
        self.results = None
        self.results_afs = None
        self.smatrix = None
        self.freq = None
        self.freq_afs = None

    def load_original(self):
        self.Substrate = Substrate.load(self.init_proj_dir)
        self.Layout = Layout.load(self.init_proj_dir)
        self.Ports = Ports.load(self.init_proj_dir)
        self.Sampling = Sampling.load(self.init_proj_dir)
        self.results = EMResults.load(self.init_proj_dir, afs=False)
        if self.Sampling.mode == 'adaptive':
            self.results_afs = EMResults.load(self.init_proj_dir, afs=True)

    def load_new(self):
        self.Substrate = Substrate.load(self.new_proj_dir)
        self.Layout = Layout.load(self.new_proj_dir)
        self.Ports = Ports.load(self.new_proj_dir)
        self.Sampling = Sampling.load(self.new_proj_dir)
        self.results = EMResults.load(self.new_proj_dir, afs=False)
        if self.Sampling.mode == 'adaptive':
            self.results_afs = EMResults.load(self.new_proj_dir, afs=True)


    def write_new(self):
        # copy_tree(self.init_proj_dir, self.new_proj_dir)
        self.Substrate.write(self.new_proj_dir)
        self.Layout.write(self.new_proj_dir)
        self.Ports.write(self.new_proj_dir)
        self.Sampling.write(self.new_proj_dir)

    # SIMULATION FUNCTIONS
    def simulate(self, folder, quiet=False):
        """
        Generate InitializationD.bat file to run simulations from current folder
        Arguments:
            :param folder: new project folder directory
            :param quiet:
        """
        print("------ RUNNING MOMENTUM SIMULATION ------")
        if self.ads_path is None:
            self.ads_path = HPEESOF_DIR
        if self.ads_license is None:
            self.ads_license = ADS_LICENSE_FILE

        import platform
        if platform.system() == 'Windows':
            # cmd = ["set HPEESOF_DIR=C:\\Agilent\\ADS2013_06\n",
            cmd = [f"set HPEESOF_DIR={self.ads_path}\n", f"set ADS_LICENSE_FILE={self.ads_license}",
                   "set PATH=%HPEESOF_DIR%\\bin;%PATH%\n", f"cd {folder}\n", "adsMomWrapper -O -3D proj proj\n"]

            file_path = os.path.join(folder, 'run.bat')
            init_run_file = open(file_path, 'w')
            init_run_file.writelines(cmd)
            init_run_file.close()
            os.system(file_path)

        elif platform.system() == 'Linux':
            folder_pathfix = folder.replace(" ", "\ ")
            cmd = ["#!/bin/bash\n",
                   f"cd {folder_pathfix}\n",
                   f"export HPEESOF_DIR={self.ads_path}\n",
                   f"export ADS_LICENSE_FILE={self.ads_license}\n",
                   f"export PATH=$HPEESOF_DIR/bin:$PATH\n",
                   f"adsMomWrapper -T proj proj\n",
                #    f"adsMomWrapper -DB proj proj\n",
                   f"adsMomWrapper -O proj proj\n",
                   f"adsMomWrapper -D -3D proj proj\n"]

            file_path = os.path.join(folder, 'run.sh')
            init_run_file = open(file_path, 'w')
            init_run_file.writelines(cmd)
            init_run_file.close()

            if quiet:
                subprocess.call(['sh', f'{file_path}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.call(['sh', f'{file_path}'], stdout=subprocess.PIPE, )
        else:
            raise ValueError("Executing from other platform")
        print("------ COMPLETED ------")

        self.results = EMResults.load(folder, afs=False)
        self.freq = np.real(np.asarray(self.results.variables[0].values))
        if self.Sampling.mode == 'adaptive':
            self.results_afs = EMResults.load(folder, afs=True)
            self.freq_afs = np.real(np.asarray(self.results_afs.variables[0].values))

        return

    def simulate_original(self):
        return self.simulate(self.init_proj_dir)

    def simulate_new(self, quiet):
        return self.simulate(self.new_proj_dir, quiet=quiet)

    # GET FREQUENCY RESPONSE FUNCTIONS
    def get_frequency_response(self, s_param, afs=False):
        if afs:
            res = self.results_afs
        else:
            res = self.results
        data_names = [d.name.lower() for d in res.data]
        idx = data_names.index(s_param.lower())
        if res.data[idx].unit == 'RI':
            s_values = np.asarray(res.data[idx].values)
        else:
            raise ValueError("ERROR! Other scattering data format to specify")
        freq = res.variables[0].values
        return freq, s_values

    def get_s11(self, afs=False):
        return self.get_frequency_response('S[1,1]', afs)

    def get_s12(self, afs=False):
        return self.get_frequency_response('S[1,2]', afs)

    def get_s21(self, afs=False):
        return self.get_frequency_response('S[2,1]', afs)

    def get_s22(self, afs=False):
        return self.get_frequency_response('S[2,2]', afs)

    def get_smatrix(self):
        f = self.results.variables[0].values
        s = np.zeros([self.Ports.number, self.Ports.number, len(f)], dtype='complex128')
        for i in range(self.Ports.number):
            for j in range(self.Ports.number):
                s[i, j] = self.get_frequency_response(f'S[{i + 1},{j + 1}]')[1]
        self.smatrix = s
        self.freq = f
        return np.asarray(f), s

    def get_smatrix_diff(self):
        pnum = int(self.Ports.number / 2)
        f = self.results.variables[0].values
        s = np.zeros([pnum, pnum, len(f)], dtype='complex128')
        for i in range(pnum):
            for j in range(pnum):
                s[i, j] = self.get_frequency_response(f'S[{i + 1},{j + 1}]')[1]
        self.smatrix = s
        self.freq = f
        return np.asarray(f).reshape(-1,1), s.reshape(-1, pnum**2)

    def get_smatrix2(self, afs=False):
        if afs:
            res = self.results_afs
            freq = self.freq_afs
        else:
            res = self.results
            freq = self.freq
        data_names = [d.name.lower() for d in res.data]
        self.smatrix = np.zeros((self.Ports.number, self.Ports.number, len(freq)), dtype=np.complex)
        self.s_list = []
        self.s_names = []
        for i in range(self.Ports.number):
            for j in range(self.Ports.number):
                s_name = f's[{i + 1},{j + 1}]'
                idx = data_names.index(s_name.lower())
                s_values = np.asarray(res.data[idx].values)
                self.s_list.append(s_values)
                self.s_names.append(s_name)
                self.smatrix[i, j] = s_values

        return freq.reshape(-1, 1), np.asarray(self.s_list).T

    def farfiels(self, freq_step, ports, voltage, impedance, visualization=1):
        ff = FarFields(self.new_proj_dir, freq_step, ports, voltage, impedance, visualization)
        run_cmd = ["#!/bin/bash\n",
                   f"cd {self.new_proj_dir}\n"
                   f"export HPEESOF_DIR={self.ads_path}\n",
                   f"export ADS_LICENSE_FILE={self.ads_licence}\n",
                   f"export PATH=$HPEESOF_DIR/bin:$PATH\n",
                   f"adsMomWrapper -FF proj proj\n"]

        file_path = os.path.join(self.new_proj_dir, 'run_ff.sh')
        run_file = open(file_path, 'w')
        run_file.writelines(run_cmd)
        run_file.close()
        os.system(f"sh {file_path}")

        ff_data = ff.load(self.new_proj_dir)

        return ff_data

    # PLOT FUNCTIONS############################################################################################
    ####################################################################################################################
    ####################################################################################################################

    def layout_plot(self, label=False, coords=False, show=True):
        # plt.figure()
        # plt.rcParams['axes.facecolor'] = 'black'
        for enum_p, p in enumerate(self.Layout.polygons):
            assert isinstance(p, Lpoly_shape), "Error! Layout polygons are corrupted "
            color = "#95d0fc"
            for mk in self.Substrate.masks:
                if mk.number == p.mask[1:]:
                    color = str('#') + mk.color[1:-1]
                    order = int(mk.number)

            order_max = max([int(mk.number) for mk in self.Substrate.masks])+1
            plt.fill(p.x_points, p.y_points, facecolor=color, edgecolor='k', zorder=order, alpha=0.6)
            plt.scatter(p.x_points, p.y_points, color='k')

            if coords:
                cnt = 0
                for x, y in zip(p.x_points, p.y_points):
                    x_ann = x - np.sign(x - p.shapely.centroid.x) * 0.1
                    y_ann = y - np.sign(y - p.shapely.centroid.y) * 0.1
                    plt.annotate(f"{p.number},{cnt}", (x_ann, y_ann), color='k', zorder=2)
                    cnt += 1
            if label:
                plt.annotate(f"P{p.number}", (p.shapely.centroid.x, p.shapely.centroid.y),
                             color='k',
                             zorder=order_max,
                             fontweight='bold',
                             fontsize=12)

        x_coord = [i[0] * 1e3 for i in self.Ports.coordinates]
        y_coord = [i[1] * 1e3 for i in self.Ports.coordinates]
        i = 0
        for xp, yp in zip(x_coord, y_coord):
            plt.annotate(f"P{i}", (xp, yp), color='k', zorder=order_max)
            i += 1
        plt.scatter(x_coord, y_coord, c='r', zorder=order_max)
        plt.axis('equal')
        plt.grid()
        legend = []
        if show:
            plt.show()

    def plot_farfields(self, ff_data, component, mode, log=False, lobes=False, layout=False):
        """
        Components order: 0: theta_angle,
                            1: phi_angle,

        Mode: 'real','imag','mag','phase'
        """
        theta = np.pi / 180 * ff_data[:, 0]  # convertion to radiants
        phi = np.pi / 180 * ff_data[:, 1]  # convertion to radiants

        if component == 0:
            field = ff_data[:, 2] + 1j * ff_data[:, 3]
        elif component == 1:
            field = ff_data[:, 4] + 1j * ff_data[:, 5]
        else:
            raise ValueError("Wrong component specified")

        if mode == 'real':
            field = np.real(field)
        elif mode == 'imag':
            field = np.imag(field)
        elif mode == 'mag':
            field = np.abs(field)
        elif mode == 'phase':
            field = np.angle(field)
        else:
            raise ValueError("Wrong mode specified")

        theta_v = theta.reshape(-1, 361)
        phi_v = phi.reshape(-1, 361)
        field = field.reshape(-1, 361)
        if log:
            field = 20 * np.log10(field)
        # field[:, -1] = field[:, 0]
        norm = colors.Normalize(vmin=np.amin(field), vmax=np.amax(field), clip=True)
        cmap = plt.cm.rainbow
        fc = cmap(norm(field))

        scale = 3
        if lobes:
            p1 = np.sin(theta_v) * np.cos(phi_v) * field * scale
            p2 = np.sin(theta_v) * np.sin(phi_v) * field * scale
            z = np.cos(theta_v) * field * scale
            shade = False

        else:
            p1 = np.sin(theta_v) * np.cos(phi_v)
            p2 = np.sin(theta_v) * np.sin(phi_v)
            z = np.cos(theta_v)
            shade = True

        fig = plt.figure(figsize=(12, 12), dpi=100)
        ax = fig.gca(projection='3d')

        if layout:
            verts = []
            x_max = 0
            y_max = 0
            for p in self.Layout.polygons:
                x_max = max([x_max, max(p.x_points)])
                y_max = max([y_max, max(p.y_points)])
            for p in self.Layout.polygons:
                assert isinstance(p, Lpoly_shape), "Error! Layout polygons are corrupted "
                color = "#95d0fc"
                for mk in self.Substrate.masks:
                    if mk.number == p.mask[1:]:
                        mc = mk.color
                        color = str('#') + mk.color
                        x_p = [i - x_max / 2 for i in p.x_points]
                        y_p = [i - y_max / 2 for i in p.y_points]
                        z_p = [-0.05 for _ in range(len(p.x_points))]
                        verts = verts + [list(zip(y_p, x_p, z_p))]

            ax.add_collection3d(Poly3DCollection(verts))

        surf = ax.plot_surface(p1, p2, z, facecolors=fc, linewidth=0, antialiased=False,
                               shade=shade)  # rstride=5, cstride=5,
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # only needed for matplotlib < 3.1
        cbar = fig.colorbar(sm)
        cbar.ax.set_xlabel('V/m')

        # ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        #
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # surf = ax.scatter(x, y, z, c=colors)
        #
        #
        # # Customize the z axis.
        # # ax.set_zlim(-1.01, 1.01)
        # # ax.zaxis.set_major_locator(LinearLocator(10))
        # # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #
        # # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

# example = Design('ex', 'ex2')
# example.load()
# example.Substrate.materials[0].permittivity = 3
# example.write()
# example.simulate()
# freq, s11 = example.get_s11()
#
# import matplotlib.pyplot as plt
# plt.plot(freq, s11)
# plt.show()
#
