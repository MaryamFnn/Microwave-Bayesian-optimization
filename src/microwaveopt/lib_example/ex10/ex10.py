import matplotlib.pyplot as plt
import numpy as np
import os
from microwaveopt.hpeesofsim.design import CircuitDesign

ex_path = __file__
ex_dir = os.path.abspath(os.path.join(ex_path, os.pardir))
init_proj = os.path.join(ex_dir, "ADS_Original_Files/MyWorkspace_wrk")
new_proj = os.path.join(ex_dir, "ADS_workdir")


device = CircuitDesign(init_proj, new_proj)
device.ads_path = '/usr/local/ADS2019'      # Initial files generated with ADS2019, use it.
device.load(init_proj)

net = device.netlist.components
net['Options'].properties['ASCII_Rawfile'] = 'yes'

net['R6'].properties['R'] = '30 Ohm'
net['C3'].properties['C'] = '0.15 pF'
net['CLin2'].properties['S'] = '92 um'
net['Tran1'].properties['StartTime'] = '1 nsec'

device.write_new()
device.simulate(new_proj, quiet=False)

t = device.results.variables['time'].data
v = device.results.variables['Vout_PAM_1'].data

plt.plot(np.asarray(t)*1e9, v)
plt.xlabel('time [nsec]')
plt.ylabel('Vout_PAM_1 [V]')
plt.show()
