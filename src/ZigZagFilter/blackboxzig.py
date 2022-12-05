from this import d
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from shapely.geometry import Polygon
from shapely import affinity
from microwaveopt.momentum.design import Design
from microwaveopt.momentum.em_setup import Sampling
from microwaveopt.em_lib import filtering
from microwaveopt.em_lib import geometry as geom

init_proj = r"C:\\Users\Administrator\Desktop\Sources\\Projects\\MicrowaveOptProject\\MyCodeForZigzagFilter\\MicrowaveOpt-new\ADS_Original_Files"
new_proj = r"C:\\Users\Administrator\Desktop\Sources\\Projects\\MicrowaveOptProject\\MyCodeForZigzagFilter\\MicrowaveOpt-new\ADS_Parametric_Layout"
device = Design(init_proj, new_proj)


L1_0 = 18
D1_0 = 0.8



def blackbox(x=None, debug=False, simm='linear', fmin = 1, fmax = 4.5):

    if x is None:
         device.load_original()
         device.layout_plot(label=True, coords=True)

    else:
        device.load_original()
        delta_L = x[0]-L1_0
        delta_D = x[1]-D1_0

# # Modifying layout
        old_geom = device.Layout.polygons
        pol = [Polygon(p.shapely.exterior.coords) for p in old_geom]
        # Changing G
        # Changing D
        pol[7] = affinity.translate(pol[7], xoff=delta_D)
        pol[5] = affinity.translate(pol[5], xoff=delta_D)
        pol[10] = affinity.translate(pol[10], xoff=delta_D)
        pol[11] = affinity.translate(pol[11], xoff=delta_D)
        pol[15] = affinity.translate(pol[15], xoff=delta_D)
        pol[14] = affinity.translate(pol[14], xoff=delta_D)
        pol[12] = affinity.translate(pol[12], xoff=delta_D)
        pol[13] = affinity.translate(pol[13], xoff=delta_D)
        pol[19] = affinity.translate(pol[19], xoff=delta_D)
        pol[18] = affinity.translate(pol[18], xoff=delta_D)


        # Changing L
        pol[5] = geom.set_dim(pol[5], delta_L, dim=1, fix=0)
        pol[6] = geom.set_dim(pol[6], delta_L, dim=1, fix=0)
        pol[9] = affinity.translate(pol[9], yoff=delta_L)
        pol[10] = affinity.translate(pol[10], yoff=delta_L)
        pol[11] = affinity.translate(pol[11], yoff=delta_L)
        pol[14] = affinity.translate(pol[14], yoff=delta_L)
        pol[15] = affinity.translate(pol[15], yoff=delta_L)
        pol[13] = affinity.translate(pol[13], yoff=delta_L)
        pol[12] = affinity.translate(pol[12], yoff=delta_L)
        pol[18] = affinity.translate(pol[18], yoff=delta_L)
        pol[19] = affinity.translate(pol[19], yoff=delta_L)



# update ports

        device.Ports.coordinates[0][0] = (pol[17].exterior.coords[0][0]) * 1e-3
        device.Ports.coordinates[0][1] = (pol[17].exterior.coords[0][1] + pol[17].exterior.coords[3][1]) * 1e-3 / 2

        device.Ports.coordinates[1][0] = (pol[18].exterior.coords[1][0]) * 1e-3
        device.Ports.coordinates[1][1] = (pol[18].exterior.coords[1][1] + pol[18].exterior.coords[2][1]) * 1e-3 / 2

        device.Layout.overwrite_geometry(pol, mask='P1')
        if debug == True:
            device.layout_plot(label=True, coords=True)

# # Modifying Simulation configuration

        if simm =='adaptive':

            device.Sampling = Sampling(mode='adaptive', lower=fmin, higher=fmax, max_samples=50)
        else:

            device.Sampling = Sampling(mode='linear', lower=fmin, higher=fmax, step=0.04)


# # run simulation
        quiet = False
        device.write_new()
        device.simulate_new(quiet=quiet)
        device.load_new()

# # modifying substrate
#     #hox_new = x[4]
#     #eox_new = x[5]
        # device.Substrate.stack[2].height = hox_0 * 1e-3
        # device.Substrate.materials[0].permittivity = eox_0


# #get data
    f, s21 = device.get_s21()
    if debug == True:
        plt.plot(f, 20*np.log10(np.abs(s21)))
        plt.show()


    return f,s21
