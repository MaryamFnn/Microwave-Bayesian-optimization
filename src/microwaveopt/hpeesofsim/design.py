import os
import platform
import subprocess

from microwaveopt.hpeesofsim.sim_res import Spectra

from microwaveopt.hpeesofsim.netlist import Netlist
from microwaveopt.config import HPEESOF_DIR, ADS_LICENSE_FILE


class CircuitDesign(object):
    """
    ADS Circuit design class
    """

    def __init__(self, folder1, folder2):
        assert os.path.isdir(folder1), "ERROR! Initial project folder does not exist"
        assert os.path.isdir(folder2), "ERROR! New project folder does not exist"

        self.ads_path = None
        self.ads_license = None

        self.init_proj_dir = folder1
        self.new_proj_dir = folder2

        self.netlist = None
        self.results = None

    def load(self, folder):
        self.netlist = Netlist.load(folder)

    def write_new(self):
        self.netlist.write(self.new_proj_dir)

    def simulate(self, folder, quiet=False):
        """ SIMULATION FUNCTION """
        print("------ RUNNING HPEESOF SIMULATION ------")
        if self.ads_path is None:
            self.ads_path = HPEESOF_DIR
        if self.ads_license is None:
            self.ads_license = ADS_LICENSE_FILE

        if platform.system() == 'Windows':
            cmd = [
                f'set SIMARCH=win32_64\n',
                f'set HOME={folder}\n',
                f'set HPEESOF_DIR={self.ads_path}\n',
                f'set COMPL_DIR=%HPEESOF_DIR%\n',
                f'set PATH=%HPEESOF_DIR%\\bin\\%SIMARCH%;%HPEESOF_DIR%\\bin;%HPEESOF_DIR%\\lib\\%SIMARCH%;%HPEESOF_DIR'
                f'%\\circuit\\lib.%SIMARCH%;%HPEESOF_DIR%\\adsptolemy\\lib.%SIMARCH%;%SVECLIENT_DIR%/bin/MATLABScript'
                f'/runtime/win64;%SVECLIENT_DIR%/sveclient;%PATH%\n',
                f'set ADS_LICENSE_FILE={self.ads_license}\n',
                f"hpeesofsim netlist.log\n",
            ]
            file_path = os.path.join(folder, 'run.bat')
            init_run_file = open(file_path, 'w')
            init_run_file.writelines(cmd)
            init_run_file.close()
            os.system(file_path)

        elif platform.system() == 'Linux':
            folder_pathfix = folder.replace(" ", "\ ")
            cmd = [
                f"#!/bin/bash\n",
                f"cd {folder_pathfix}\n",
                f"export SIMARCH=linux_x86_64\n",
                f"export HOME={folder_pathfix}\n",
                f"export HPEESOF_DIR={self.ads_path}\n",
                f"export COMPL_DIR=$HPEESOF_DIR\n",
                f"export LD_LIBRARY_PATH=$HPEESOF_DIR/lib/linux_x86_64:$HPEESOF_DIR/adsptolemy/lib.linux_x86_64"
                f":$HPEESOF_DIR/SystemVue/2014.10/linux_x86_64/lib\n",
                f"export PATH=$HPEESOF_DIR/bin/$SIMARCH:$HPEESOF_DIR/bin:$HPEESOF_DIR/lib/$SIMARCH:$HPEESOF_DIR"
                f"/circuit/lib.$SIMARCH:$HPEESOF_DIR/adsptolemy/lib.$SIMARCH:$SVECLIENT_DIR/bin/MATLABScript/runtime"
                f"/win64:$SVECLIENT_DIR/sveclient:$PATH\n",
                f"export ADS_LICENSE_FILE={self.ads_license}\n",
                f"hpeesofsim netlist.log\n",
            ]

            file_path = os.path.join(folder, 'run.sh')
            init_run_file = open(file_path, 'w')
            init_run_file.writelines(cmd)
            init_run_file.close()

            if quiet:
                subprocess.call(['sh', f'{file_path}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                file_path = file_path.replace(" ", "\ ")
                os.system('sh ' + file_path)
                # subprocess.call(['sh', f'{file_path}'], stdout=subprocess.PIPE, )


            self.results = Spectra.load(folder)


        else:
            raise ValueError("Executing from other platform")

        print("------ COMPLETED HPEESOF SIMULATION ------")

        return
