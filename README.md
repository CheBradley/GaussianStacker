# GaussianStacker

GaussianStacker is designed to stack FITS images taken from the FIRST or Deep VLA surveys and fit a 2D gaussian to the image. The code first generates and idealized gaussian and then uses curve_fit to fit that gaussian to the data. 

The code can create a figure to show each image being stacked and/or create a pdf containing those figures. For large datasets, opening a lot of figures is not recommended. 


