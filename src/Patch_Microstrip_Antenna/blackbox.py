
import os
import numpy as np
from microwaveopt.momentum.design import Design
import matplotlib.pyplot as plt
from microwaveopt.momentum.em_setup import Sampling
import shapely
from microwaveopt.em_lib import geometry as geom
from microwaveopt.em_lib import filtering

init_proj = r"C:\\Users\\Administrator\\Desktop\\Sources\\MicrowaveOptProject\\MyCodeForAntennaPatch\\MicrowaveOpt-new\\microwaveopt\\lib_example\\ADS_ex\\PatchAntenna_ex2_org"
new_proj = r"C:\\Users\\Administrator\\Desktop\\Sources\\MicrowaveOptProject\\MyCodeForAntennaPatch\\MicrowaveOpt-new\\microwaveopt\\lib_example\\ADS_ex\\PatchAntenna_ex2_param"
device = Design(init_proj, new_proj)

# INITIAL DESIGN PARAMETERS:
w1_0 = 35
w2_0 = 2.9
l2_0 = 8.7
l1_0 = 30.5
hox_0 = 1.5
eox_0 = 4.3


def blackbox(x=None, debug=False, simm='linear', fmin = 2, fmax = 3):

    if x is None:
         x = [w1_0, l1_0]
         print(x)
        #  device.simulate_original()
         device.load_original()
         device.layout_plot(label=True, coords=True)

    else:
        device.load_original()
    #device.layout_plot(label=True, coords=True)

        delta_w = x[0]-w1_0
        delta_l = x[1]-l1_0

# # Modifying layout
        polygons = device.Layout.shapely()
        polygons[1] = geom.set_dim(polygons[1], delta_w/2 ,0,fix=1)
        polygons[3] = geom.set_dim(polygons[3], delta_w/2,0 , fix=0)
        polygons[1] = geom.set_dim(polygons[1], delta_l,1, fix=0)
        polygons[2] = geom.set_dim(polygons[2], delta_l, 1,fix=0)
        polygons[3] = geom.set_dim(polygons[3], delta_l, 1,fix=0)
        device.Layout.overwrite_geometry(polygons, mask='P1')
        if debug == True:
            device.layout_plot(label=True, coords=True)

# update ports

        device.Ports.coordinates[0][0] = (polygons[0].exterior.coords[0][0] + polygons[0].exterior.coords[1][0]) * 1e-3 / 2 # covertion to mm
        device.Ports.coordinates[0][1] = (polygons[0].exterior.coords[0][1]) * 1e-3

# # modifying substrate
#     #hox_new = x[4]
#     #eox_new = x[5]
        device.Substrate.stack[2].height = hox_0 * 1e-3
        device.Substrate.materials[0].permittivity = eox_0


# # Modifying Simulation configuration

        if simm =='adaptive':

            device.Sampling = Sampling(mode='adaptive', lower=fmin, higher=fmax, max_samples=50)
        else:
            device.Sampling = Sampling(mode='linear', lower=fmin, higher=fmax, step=0.04)

# # run simulation

        device.write_new()
        device.simulate_new(quiet=not debug)
        device.load_new()

# #get data
    f, s11 = device.get_s11()
    s11_db = 20*np.log10(np.abs(s11))
    bw = filtering.bandwidth(f, s11_db, method=1)
    f0 = filtering.central_frequency(f, s11_db, method=1)
    if debug == True:
        plt.plot(f, 20*np.log10(np.abs(s11)))
        plt.show()


    return f,s11,f0,bw

