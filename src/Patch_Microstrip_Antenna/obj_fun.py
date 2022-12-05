
import blackbox
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def obj_fun(x=[None], debug=False):   ## x is the list of our parameters
    y = []

    for i, xi in enumerate(x):
        f ,s11,f0,bw = blackbox.blackbox(xi, debug=debug, simm='linear',fmin = 2 , fmax = 3)
        plt.plot(f, 20*np.log10(np.abs(s11)))
        plt.savefig(f".\\figures\\fig.png")
        dist = - np.abs(f0/1e9 - 2.5) # to minimize distance between f0 and 2.4 GHz
        with open ("log.txt",'a') as O:
            O.write(f"f0 = {f0/1e9}\n")
            O.write(f"bw = {bw/1e9}\n")
            O.write(f"x={xi}\n")
            O.write(f"y={dist}\n----------------\n")
        y.append(dist)
        plt.plot(i,dist)
        plt.savefig(f".\\figures\\fig1.png")
    return np.vstack(y)


if __name__=='__main__':
    obj_fun(debug=True)