#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import numpy as np
# from scipy.optimize import curve_fit
import numpy as nnp
import cupy as cp
import cunumeric as np
import jax.numpy as jnp
import scipy
from jaxfit import CurveFit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from dlpack import asdlpack

from legate.core.task import task, InputStore, OutputStore
from legate.core._ext.task.util import KNOWN_VARIANTS
from legate.core import (
    align,
    get_legate_runtime,
    LegateDataInterface,
    LogicalStore,
    StoreTarget,
    # VariantCode,  # TODO: introduced later
)

def get_store(obj: LegateDataInterface) -> LogicalStore:
    iface = obj.__legate_data_interface__
    assert iface["version"] == 1
    data = iface["data"]
    # There should only be one field
    assert len(data) == 1
    field = next(iter(data))
    assert not field.nullable
    column = data[field]
    assert not column.nullable
    return column.data

# Our function to fit is going to be a sum of two-dimensional Gaussians
def gaussian_jax(x, y, x0, y0, xalpha, yalpha, A, delta):
    delta = delta*jnp.pi/180
    xp = (x-x0)*jnp.cos(delta) + (y-y0)*jnp.sin(delta)
    yp = -(x-x0)*jnp.sin(delta) + (y-y0)*jnp.cos(delta)
    return A * jnp.exp( -(xp/xalpha)**2 -(yp/yalpha)**2)

def gaussian(x, y, x0, y0, xalpha, yalpha, A, delta):
    delta = delta*np.pi/180
    xp = (x-x0)*np.cos(delta) + (y-y0)*np.sin(delta)
    yp = -(x-x0)*np.sin(delta) + (y-y0)*np.cos(delta)
    return A * np.exp( -(xp/xalpha)**2 -(yp/yalpha)**2)

@task(variants=tuple(KNOWN_VARIANTS))
def curve_fit_wrapper(xdata: InputStore,
                      ydata: InputStore,
                      # p0: list,
                      # bounds: list,
                      popt: OutputStore) -> None:

    # This is the callable that is passed to curve_fit. M is a (2,N) array
    # where N is the total number of data points in Z, which will be ravelled
    # to one dimension.
    def _gaussian(M, *args):
        x, y = M
        return gaussian_jax(x,y,*args)

    p0 = (0, 0, 1, 1, 2, 30)
    bounds=([-np.inf,-np.inf,0,0,0,0],
            [np.inf,np.inf,np.inf,np.inf,np.inf,360])

    cf = CurveFit()

    if xdata.target == StoreTarget.FBMEM or xdata.target == StoreTarget.ZCMEM:
        xdata_arr = cp.asarray(xdata.get_inline_allocation())
        ydata_arr = cp.asarray(ydata.get_inline_allocation())
        popt_output, pcov = cf.curve_fit(_gaussian, xdata_arr, ydata_arr, p0=p0, bounds=bounds)
        popt_arr = cp.asarray(popt.get_inline_allocation())
        popt_arr[:] = cp.asarray(popt_output)
    else:
        xdata_arr = nnp.asarray(xdata.get_inline_allocation())
        ydata_arr = nnp.asarray(ydata.get_inline_allocation())
        popt_output, pcov = cf.curve_fit(_gaussian, xdata_arr, ydata_arr, p0=p0, bounds=bounds)
        popt_arr = nnp.asarray(popt.get_inline_allocation())
        popt_arr[:] = popt_output

def curve_fit(xdata, ydata, p0, bounds, popt):
    curve_fit_wrapper(get_store(xdata),
                      get_store(ydata),
                      # p0, bounds,
                      get_store(popt))

# The two-dimensional domain of the fit.
N = 256
M = 256
y = np.linspace(-N/2,N/2-1, N)
x = np.linspace(-M/2,M/2-1, M)
X,Y = np.meshgrid(x, y)


# A list of the Gaussian parameters: x0, y0, xalpha, yalpha, A, delta
gprms = (0, 2, 15, 5.4, 3, 25)

# Standard deviation of normally-distributed noise to add in generating
# our test function to fit.
nnp.random.seed(0)
noise_sigma = 0.1

# The function to be fit is Z.
Z = gaussian(X, Y, *gprms)
Z += noise_sigma * nnp.random.randn(*Z.shape)

# Plot the 3D figure of the fitted function and the residuals.
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, cmap='plasma')
# ax.set_zlim(0,np.max(Z)+2)
# plt.show()
plt.figure()
# plt.imshow(Z)
plt.savefig("z.png")

# In[14]:


# Initial guesses to the fit parameters.
p0 = [0, 0, 1, 1, 2, 30]
# Flatten the initial guess parameter list.
# p0 = [p for prms in guess_prms for p in prms]

# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
xdata = np.vstack((X.ravel(), Y.ravel()))
popt = np.zeros(len(p0))
# Do the fit, using our custom _gaussian function which understands our
# flattened (ravelled) ordering of the data points.
curve_fit(xdata, Z.ravel(),
          p0,
          bounds=([-np.inf,-np.inf,0,0,0,0],
                  [np.inf,np.inf,np.inf,np.inf,np.inf,360]),
          popt=popt)
fit = gaussian(X,Y,*popt)

# fit = np.zeros(Z.shape)
# for i in range(len(popt)//6):
#     fit += gaussian(X, Y, *popt[i*6:i*6+6])
print('Fitted parameters:')
print(popt)

rms = np.sqrt(np.mean((Z - fit)**2))
print('RMS residual =', rms)

# Plot the 3D figure of the fitted function and the residuals.
# fig = plt.figure()
# # ax = fig.gca(projection='3d')
# ax = fig.gca()
# ax.plot_surface(X, Y, fit, cmap='plasma')
# cset = ax.contourf(X, Y, Z-fit, zdir='z', offset=-4, cmap='plasma')
# ax.set_zlim(-4,np.max(fit))
# plt.savefig('plt.png')

# plt.figure()
# # plt.imshow(fit)
# plt.savefig('fit.png')
# # Plot the test data as a 2D image and the fit as overlaid contours.
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.imshow(Z, origin='bottom', cmap='plasma',
# #           extent=(x.min(), x.max(), y.min(), y.max()))
# # ax.contour(X, Y, fit, colors='w')
# # plt.show()


# # In[11]:


# popt


# # In[ ]:
