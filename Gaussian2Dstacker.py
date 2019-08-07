import astropy
import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy

from matplotlib import colors, cm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting
from numpy import zeros
from scipy import optimize
from sherpa.astro.ui import *
from astropy import visualization
from astropy.io import fits
from mpl_toolkits.mplot3d import Axes3D
plt.style.use(visualization.astropy_mpl_style)

folder = input("What is the name of the folder with the images?: ")
sys.path.append("/" + folder)
files = sorted(os.listdir(folder))
showimages = input('Show individual images? (Not recommended for large datasets): ')
if showimages == 'y' or showimages == 'yes' or showimages == 'Yes':
	showimages == 'y'
else:
	showimages == 'n'

createpdf = input('Create a pdf with all of the figures?: ')
if createpdf == 'y' or createpdf == 'yes' or createpdf == 'Yes':
	createpdf = 'y'
if createpdf == 'y' or showimages == 'y':
	print('Images per page - Large:25, Medium:49, Small:64')
	size = input('Would you like small, medium, or large images?: ')
	if size == 's' or size == 'small' or size == 'Small':
		images = 64
	elif size == 'm' or size == 'medium' or size == 'Medium':
		images = 49
	elif size == 'l' or size =='large' or size == 'Large':
		images = 25
print('FIRST or Deep VLA images?')
catalog = input('Type 1 for FIRST or 2 for Deep VLA: ')
if catalog == '1':
	sigmaguess = 2.123
if catalog == '2':
	sigmaguess = 0.764

def gaussianfit(x,y,*args):
	"""
	This function makes an idealized gaussian with the given parameters,
	then returns the value of that gaussian at a specific point or points
	"""
	g_init1 = models.Gaussian2D(*args[0:5])
	if len(args)>5:
		z = g_init1(x,y) 
	else:
		z = g_init1(x,y)
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


def fitgaussian(matrix, bounds):
	"""
	This is the main function. It takes a matrix and, using the other 
	functions, fits a 2D gaussian to it. It then returns that fit, the 
	optimized parameters, the covariance of the parameters, and two variables that contain information on 
	the size of the matrix
	"""
	Y, X = np.indices(matrix.shape)
	sizeim = len(matrix)
	p0 = [np.max(matrix[13:20,13:20]),sizeim/2,sizeim/2,sigmaguess,sigmaguess]
	
	# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
	xdata = np.vstack((X.ravel(), Y.ravel()))
	
	popt, pcov = curve_fit(_gaussian, xdata, matrix.ravel(), p0, bounds=bounds, maxfev=50000)
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

#removes all nan and infs
badvals = []
for imnum in range(numQSOs): 
	matrix = QSO[imnum,:,:]
	if np.any(np.isnan(matrix)):
		badvals.append(imnum)
	elif np.any(np.isinf(matrix)):
		badvals.append(imnum)

QSO = np.delete(QSO,badvals,0)
			
numQSOs = len(QSO)

bounds = ([0, sizeim/2-0.5,sizeim/2-0.5,0,0], [sizeim, sizeim/2+0.5,sizeim/2+0.5, 3,3]) 			
if createpdf == 'y':
	if numQSOs > images:
		sizeplot = np.sqrt(images)
	else:
		sizeplot = math.ceil(math.sqrt(numQSOs))
	with PdfPages(folder+'.pdf') as pdf:
		for imnum in range(numQSOs):
			cutoff = (imnum+1)%images
			if cutoff == 1:
				allfigs = plt.figure()
			if cutoff == 0:
				cutoff = images
			plt.subplots_adjust(wspace=0, hspace=0.5, right = .95, left = .05, top = .9, bottom = .1)
			axis = allfigs.add_subplot(sizeplot,sizeplot,cutoff)
			axis.set_xticklabels([]) #hide labels
			axis.set_yticklabels([])
			axis.grid(False) #hide gridlines
			axis.imshow(QSO[imnum,:,:])
			name = str(files[imnum])
			name = name.replace('.fits', '')
			axis.set_title(name + ' ('+str(imnum+1)+')', size=4)
			if cutoff == images or imnum == numQSOs-1:
				pdf.savefig(allfigs)
			if showimages == 'n':
				plt.close("all")

if showimages == 'y':
	if numQSOs > images:
		sizeplot = np.sqrt(images)
		print('sizeplot = ',sizeplot)
	else:
		sizeplot = math.ceil(math.sqrt(numQSOs))
	for imnum in range(numQSOs):
		cutoff = (imnum+1)%images
		if cutoff == 1:
			allfigs = plt.figure()
		if cutoff == 0:
			cutoff = images
		plt.subplots_adjust(wspace=0, hspace=0.5, right = .9, left = .1, top = .9, bottom = .1)
		axis = allfigs.add_subplot(sizeplot,sizeplot,cutoff)
		axis.set_xticklabels([]) #hide labels
		axis.set_yticklabels([])
		axis.grid(False) #hide gridlines
		axis.imshow(QSO[imnum,:,:])
		name = str(files[imnum])
		name = name.replace('.fits', '')
		axis.set_title(name + ' ('+str(imnum+1)+')', size=4)

stacked_matrix = np.median(QSO, axis = 0)

stacked_output = fitgaussian(stacked_matrix, bounds)
fitted_stacked_gaussian = stacked_output[0]
(amplitude, x, y, sigma_x, sigma_y) = stacked_output[1]
X = stacked_output[2]
Y = stacked_output[3]
pcov = stacked_output[4]

stacked_fig = plt.figure()
axstacked = stacked_fig.add_subplot(111)
axstacked.imshow(stacked_matrix, origin='bottom', cmap='plasma')
axstacked.contour(X, Y, fitted_stacked_gaussian, colors='w')

stacked3Dfig = plt.figure()
ax3D = stacked3Dfig.gca(projection='3d')
ax3D.plot_surface(X, Y, stacked_matrix, cmap = 'gist_heat')
ax3D.plot_surface(X, Y, fitted_stacked_gaussian, cmap='plasma')

I = amplitude*sigma_x*sigma_y*2*math.pi 
(ampcov, x_cov, y_cov, sigma_x_cov, sigma_y_cov) = np.diag(pcov)
perr = np.sqrt(np.diag(pcov))
param1 = (ampcov/amplitude)*(ampcov/amplitude)
param2 = (sigma_x_cov/sigma_x)*(sigma_x_cov/sigma_x)
param3 = (sigma_y_cov/sigma_y)*(sigma_y_cov/sigma_y)
Icov = I * np.sqrt(param1+param2+param3)

xdata = np.vstack((X.ravel(), Y.ravel()))

if sizeim%2==0:
	lowerbound = sizeim/2-0.5
	upperbound = sizeim/2+0.5
else:
	lowerbound = math.floor(sizeim/2)
	upperbound = math.ceil(sizeim/2)
	
nbounds = [[0,lowerbound,lowerbound,0,0],[sizeim,upperbound,upperbound,3,3]]

newstackedoutput = fitgaussian(stacked_matrix, nbounds)
new_fitted_gaussian = newstackedoutput[0] + Icov
(amplitude, x, y, sigma_x, sigma_y) = newstackedoutput[1]
newpcov = newstackedoutput[4]

I = amplitude*sigma_x*sigma_y*2*math.pi 
(ampcov, x_cov, y_cov, sigma_x_cov, sigma_y_cov) = np.diag(newpcov)
perr = np.sqrt(np.diag(newpcov))
param1 = (ampcov/amplitude)*(ampcov/amplitude)
param2 = (sigma_x_cov/sigma_x)*(sigma_x_cov/sigma_x)
param3 = (sigma_y_cov/sigma_y)*(sigma_y_cov/sigma_y)
Icov = I * np.sqrt(param1+param2+param3)

print('x           = ',x)
print('y           = ',y)
print('sigma_x     = ',sigma_x)
print('sigma_x_cov = ',sigma_x_cov)
print('simga_y     = ',sigma_y)
print('sigma_y_cov = ',sigma_y_cov)
print('amplitude   = ',amplitude)
print('ampcov err  = ',perr[0])
print('Integral    = ',I)
print('Icov        = ',Icov)
print('I +- Icov   = ',I+Icov, I-Icov)
newfig = plt.figure()
newax = newfig.add_subplot(111)

newax.imshow(stacked_matrix, cmap = 'plasma')
newax.contour(X, Y, new_fitted_gaussian, colors='w')

new3Dfig = plt.figure()
ax = new3Dfig.gca(projection='3d')
ax.plot_surface(X, Y, stacked_matrix, cmap = 'gist_heat', alpha = 0.5)
ax.plot_surface(X, Y, new_fitted_gaussian,  cmap='plasma')
ax.set_zlim(stacked_matrix.min(),np.max(fitted_stacked_gaussian)+np.min(fitted_stacked_gaussian))

oneimage = plt.figure()
oneaxis = oneimage.add_subplot(111)

oneaxis.imshow(stacked_matrix, cmap='gist_heat')
oneaxis.grid(False)
oneaxis.set_xticklabels([]) #hide labels
oneaxis.set_yticklabels([])

#creates a fits file from the stacked gaussian
hdu = fits.PrimaryHDU(new_fitted_gaussian)
hdu.writeto(folder + "stacked.fits")

plt.show()
