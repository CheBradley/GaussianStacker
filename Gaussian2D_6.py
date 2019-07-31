import astropy
import sys
import os
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import scipy.stats as stats
import scipy

from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting
from numpy import zeros
from scipy import optimize
from sherpa.astro.ui import *
from astropy import visualization
from astropy.io import fits
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.style.use(visualization.astropy_mpl_style)
np.set_printoptions(threshold=sys.maxsize) #allows me to see all the data without truncation
#np.set_printoptions(threshold=10)

folder = input("What is the name of the folder with the images?: ")
sys.path.append("/" + folder)
files = sorted(os.listdir(folder))

def gaussianfit(x,y,amplitude,x_mean,y_mean,x_stddev,y_stddev,const):
	"""
	This function makes an idealized gaussian with the given parameters,
	then returns the value of that gaussian at a specific point or points
	"""
	g_init = models.Gaussian2D(amplitude, x_mean, y_mean, x_stddev, y_stddev)
	z = g_init(x,y) + const
	return z

# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def _gaussian(M, *args):
	"""
	This is the callable that is passed to curve_fit. M is a (2,N) array,
	where N is the total number of data points in Z, which will be ravelled
	to one dimension.
	"""
	x, y = M
	array = np.zeros(x.shape)
	array += gaussianfit(x, y, *args)
	return array 


def fitgaussian(matrix, guess_prms):
	"""
	This is the main function. It takes a matrix and, using the other 
	functions fits a 2D gaussian to it. It then returns that fit, the 
	optimized parameters, and two variables that contain information on 
	the size of the matrix
	"""
	Y, X = np.indices(matrix.shape)
	sizeim = len(matrix)
	p0 = guess_prms
	const = guess_prms[5]
	
	# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
	xdata = np.vstack((X.ravel(), Y.ravel()))
	
	# Do the fit, using our custom _gaussian function which understands our
	# flattened (ravelled) ordering of the data points.
	popt, pcov = curve_fit(_gaussian, xdata, matrix.ravel(), p0, bounds=(0, [sizeim, sizeim/2+1,sizeim/2+1, 3,3, matrix.max()]) ,maxfev=50000) 
	fitted_gaussian = gaussianfit(X, Y, *popt)
	print('Fitted parameters:')
	print(popt)
	return (fitted_gaussian, popt, X, Y, pcov)
	

numQSOs = len(files)

#gets the size of an image so it knows how large to make the matrix 
sizeim = len(fits.getdata(folder+"/"+files[0])) 
QSO = zeros([numQSOs, sizeim, sizeim])

for QSOnum in range(numQSOs):
	data = fits.getdata(folder+"/"+files[QSOnum])
	QSO[QSOnum,:,:] = data


#you can open files or get their data
#im not sure what opening does, but you want to use fits.getdata

badvals = []
for imnum in range(numQSOs): 
	matrix = QSO[imnum,:,:]
	if np.any(np.isnan(matrix)):
		badvals.append(imnum)
	elif np.any(np.isinf(matrix)):
		badvals.append(imnum)

QSO = np.delete(QSO,badvals,0)
		
numQSOs = len(QSO)

if numQSOs > 64:
	sizeplot = 8
	numfigs = math.ceil(numQSOs/64)
else:
	numfigs = 1
	sizeplot = math.ceil(math.sqrt(numQSOs))
	

for imnum in range(numQSOs):
	cutoff = (imnum+1)%64
	if cutoff == 1:
		plt.figure()
	if cutoff == 0:
		cutoff = 64
	plt.subplot(sizeplot,sizeplot,cutoff)
	plt.imshow(QSO[imnum,:,:])
	name = str(files[imnum])
	name = name.replace('.fits', '')
	plt.gca().set_title(name + '('+str(imnum)+')', fontsize = 6)

guess_prms = [.001, sizeim/2, sizeim/2, 1.8, 1.8, 0]
for imnum in range(numQSOs):
	cutoff = (imnum+1)%64
	if cutoff == 1:
		figs_gaussian = plt.figure()
		#figs_3D = plt.figure()
	if cutoff == 0:
		cutoff = 64
	matrix = QSO[imnum,:,:]
	name = str(files[imnum])
	name = name.replace('.fits', '')
	print(name + '('+str(imnum)+')')
	output = fitgaussian(matrix, guess_prms)
	fitted_gaussian = output[0]
	popt = output[1]
	X = output[2]
	Y = output[3]
	pcov = output[4]
	
	# Plot the test data as a 2D image and the fit as overlaid contours.
	axgauss = figs_gaussian.add_subplot(sizeplot,sizeplot,cutoff)
	axgauss.imshow(matrix, origin='bottom', cmap='plasma')
	axgauss.contour(X, Y, fitted_gaussian, colors='w')


stacked_matrix = np.median(QSO, axis = 0)

plt.figure()
plt.contour(stacked_matrix)

stacked_output = fitgaussian(stacked_matrix,guess_prms)
fitted_stacked_gaussian = stacked_output[0]
(amplitude, x, y, sigma_x, sigma_y, const) = stacked_output[1]
X = stacked_output[2]
Y = stacked_output[3]
pcov = stacked_output[4]

stacked_fig = plt.figure()
axstacked = stacked_fig.add_subplot(111)
axstacked.imshow(stacked_matrix, origin='bottom', cmap='plasma')
axstacked.contour(X, Y, fitted_stacked_gaussian, colors='w')


I = amplitude*sigma_x*sigma_y*2*math.pi 
(ampcov, x_cov, y_cov, sigma_x_cov, sigma_y_cov, const_cov) = np.diag(pcov)
perr = np.sqrt(np.diag(pcov))
param1 = (ampcov/amplitude)*(ampcov/amplitude)
param2 = (sigma_x_cov/sigma_x)*(sigma_x_cov/sigma_x)
param3 = (sigma_y_cov/sigma_y)*(sigma_y_cov/sigma_y)
Icov = I * np.sqrt(param1+param2+param3)

print('amplitude  = ',amplitude)
print('ampcov err = ',perr[0])
print('Integral   = ',I)
print('Icov       = ',Icov)
print('I +- Icov  = ',I+Icov, I-Icov)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, fitted_stacked_gaussian, cmap='plasma')
ax.set_zlim(0,np.max(fitted_stacked_gaussian)+np.min(fitted_stacked_gaussian))


#~ hdu = fits.PrimaryHDU(fitted_stacked_gaussian)
#~ hdu.writeto("gausstester2.fits")
plt.show()
