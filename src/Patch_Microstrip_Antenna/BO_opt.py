import blackbox
import obj_fun
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
##---------------------------------------------
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf


#create initial data (10 pairs of parameters)
init_sample_num =3
input_dim = 2
bounds = np.array([[32,38],[27.5,33.5]])
sampler = qmc.LatinHypercube(d=input_dim)
init_points_normalized = sampler.random(n=init_sample_num)
init_points = (init_points_normalized * (bounds[:,1] - bounds[:,0]).T) + bounds[:,0].T
print(init_points)



#get the data of these initial samples
init_values =obj_fun.obj_fun(x=init_points, debug=False)
#Save initial samples
init_samples = np.hstack((init_points, init_values))
np.save("ex_opt_init.npy", init_samples)


###### GP section

train_X = torch.Tensor(init_samples[:,:-1])
train_Y = torch.Tensor(init_samples[:,-1].reshape(-1,1))
print(train_X)
print(train_Y)
iteretion =2;
for i in range(iteretion):

    count =i + 1

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


# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection='3d')
# zdata=train_Y
# xdata=train_X.numpy()[0]
# ydata=train_X.numpy()[1]
# ax.scatter3D(xdata, ydata, zdata, c=zdata)
# plt.savefig(f".\\figures\\fig3d.png")
######finding best solution
print('hiiii')
y_opt_idx = np.argmax(train_Y.numpy())
y_opt= np.max(train_Y.numpy())
x_opt = train_X.numpy()[y_opt_idx]
print(x_opt)
#c = obj_fun.obj_fun(x=x_opt)
f ,s11,f0,bw = blackbox.blackbox(x_opt, debug=True, simm='linear',fmin = 2 , fmax = 5)
plt.plot(f, 20*np.log10(np.abs(s11)))
plt.savefig(f".\\figures\\fig1.png")
print(f"Best solution found: x={x_opt}, y={y_opt}")