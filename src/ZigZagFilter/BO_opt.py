from turtle import color
from xmlrpc.server import list_public_methods

from matplotlib import cm
import blackboxzig
import obj_fun
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
##---------------------------------------------
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from math import dist
from scipy.interpolate import griddata
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
######Initial responces block######

#create initial data (10 pairs of parameters)
init_sample_num =3
input_dim = 2
bounds = np.array([[5,25],[0.3,1.2]])
sampler = qmc.LatinHypercube(d=input_dim)
init_points_normalized = sampler.random(n=init_sample_num)
init_points = (init_points_normalized * (bounds[:,1] - bounds[:,0]).T) + bounds[:,0].T
print(init_points)


###### Objective function block#####

#get the data of these initial samples
init_values =obj_fun.obj_fun(x=init_points, debug=False)
#Save initial samples
init_samples = np.hstack((init_points, init_values))
np.save("ex_opt_init.npy", init_samples)


###### GP model block ######


train_X = torch.Tensor(init_samples[:,:-1])
train_Y = torch.Tensor(init_samples[:,-1].reshape(-1,1))


print(train_X.shape)
print(train_Y.shape)
iteretion =3;
ph=[25,1.2]
pl=[5,.3]
ps=[18.678,0.79]
regret=[]
counter=[]
for i in range(iteretion):

    count =i

    train_Y_norm = (train_Y - train_Y.mean()) / train_Y.std() # normalize values
    gp = SingleTaskGP(train_X, train_Y_norm)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)


# Compute acquisition function
    from botorch.acquisition import UpperConfidenceBound
    UCB = UpperConfidenceBound(gp, beta=0.1)

    # Maximize acquisition function
    acq_bounds = torch.Tensor(bounds.T)  # convert input bounds for the acquisition function
    next_point, acq_value = optimize_acqf(UCB, bounds=acq_bounds, q=1, num_restarts=5, raw_samples=20,)

    # Compute objective values at the next candidate point
    next_value = obj_fun.obj_fun(next_point, debug=False)
    next_value = torch.Tensor(next_value)

    # Add next sample to the dataset
    train_X = torch.vstack((train_X, next_point))
    train_Y = torch.vstack((train_Y, next_value))
    print (count)

    #Regret
    y_opt_idx = np.argmax(train_Y.numpy())
    y_opt= np.max(train_Y.numpy())
    x_opt = train_X.numpy()[y_opt_idx]
    regret.append(dist(x_opt,ps)/dist(ph,pl))
    counter.append(count)

plt.figure()
plt.plot(counter,regret)
plt.savefig(f".\\figures\\Regret1.png")

#prediction
##create new inputs
# init_sample_num =1000
# input_dim = 2
# bounds = np.array([[5,25],[0.3,1.2]])
# sampler = qmc.LatinHypercube(d=input_dim)
# init_points_normalized = sampler.random(n=init_sample_num)
# init_points = (init_points_normalized * (bounds[:,1] - bounds[:,0]).T) + bounds[:,0].T
# test_X = torch.Tensor(init_points)
# # a=gp.forward(test_X)

# gp.eval()
# posterior = gp.posterior(test_X)
# a=posterior.mean.cpu()
# a = a.detach().numpy()

# G_grid = np.linspace(5,25,200)
# L_grid = np.linspace(.3,1.2,200)
# G_mesh,L_mesh = np.meshgrid(G_grid,L_grid)


# z_grid = griddata(test_X,a.flatten(),(G_mesh,L_mesh), method = 'linear')

# # xtestarr=test_X[:,0].numpy()
# # ytestarr=test_X[:,1].numpy()

# fig = plt.figure()
# ax =plt.axes(projection='3d', computed_zorder=False)
# ax.plot_surface(G_mesh,L_mesh,z_grid,cmap=cm.coolwarm, alpha=0.8,zorder=-1)
# ax.scatter( train_X[:,0], train_X[:,1], train_Y,color='k')
# plt.savefig(f".\\figures\\fig4.png")



#finding the best solution
y_opt_idx = np.argmax(train_Y.numpy())
y_opt= np.max(train_Y.numpy())
x_opt = train_X.numpy()[y_opt_idx]
f ,s21 = blackboxzig.blackbox(x_opt, debug=True, simm='linear',fmin = 1 , fmax = 4)
plt.plot(f, 20*np.log10(np.abs(s21)))
plt.savefig(f".\\figures\\BestSolution.png")
indexes= f.index([2.45e9,2.55e9])
values=s21(indexes)
indexess21=s21.index(.708)
valusef=f(indexess21)
with open ("BestSolution.txt",'a') as O:
            O.write(f"Best solution found: x={x_opt}, y={y_opt}")


