import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.ndimage as ndimage
import scipy.signal as signal

import math
import hough



# Attempt to write a general echelle sepctra routine


# 3/29/2018 created 

# Used to extract echelle orders:
# inputs:
#   externalData: 2x2 array of echelle spectra pixel values

#   optional:
#     medianFilterShape: extent of median filter to be applied in preprocessing
#                        (2,1) becomes a window of shape (5,3)
def extractOrders(externalData, parameter_arrays, 
                  medianFilterShape=(2,1), verbose=True,
                  spacing = 10, sigma = 3, 
                  min_peak_size = 1, max_peak_size=50,
                  include_border=False,
                  cutoff_fraction=2.0, neighborhood_size=40):
  #Check externalData has right shape:
  if(len(np.shape(externalData)) == 2):
    pass
  else:
    raise(TypeError, "Input data is not 2x2 array. If erroneous, try converting input to numpy array.")


  # Copy externalData by value so we don't alter it
  # data is a value-copy of externalData
  data = []
  data.extend(externalData)
  data = np.array(data)

  if verbose:
    print("Preprocessing")

  #Preprocess data:
  # 1) apply median filter 

  windowShape = (medianFilterShape[0]*2+1, medianFilterShape[1]*2+1)
  data = ndimage.median_filter(data, size=windowShape)


  if verbose:
    print("Finding Peaks")
  #find peaks column wise
  peaks,bounds = get1Dpeaks(data, spacing=spacing, sigma=sigma, min_peak_size=min_peak_size, max_peak_size=max_peak_size, include_border=include_border)

  # Hough Transform:
  # y = ax**2 + bx + c


  if verbose:
    print("Performing Hough Transform on above values")
  above_peaks = (bounds[0], peaks[1])
  above_HT = hough.parabolicHT(above_peaks, parameter_arrays,verbose=verbose)

  if verbose:
    print("Performing Hough Transform on below values")

  below_peaks = (bounds[1], peaks[1])
  below_HT = hough.parabolicHT(below_peaks, parameter_arrays,verbose=verbose)

  if verbose:
    print("Seaching for Peaks")
  above_coordinates = hough.findThreePeaks(above_HT, cutoff_fraction=cutoff_fraction, neighborhood_size = neighborhood_size)

  below_coordinates = hough.findThreePeaks(below_HT, cutoff_fraction=cutoff_fraction, neighborhood_size = neighborhood_size)


  if verbose:
    print("Plotting")
  above_x,above_ys = genThreeParabola(above_coordinates, parameter_arrays, data)

  below_x,below_ys = genThreeParabola(below_coordinates, parameter_arrays, data)

  return above_x,(above_ys,below_ys)

  # plt.imshow(data,origin='lower')
  # plt.scatter(peaks[1],peaks[0])

  # for y in ys:
  #   plt.plot(x,y,color='k')
  # plt.show()

  # return peaks


#### This function selects peaks column-wise in data
#    Used to select center of echelle orders


#    Output: rows,cols -- arrays corresponding to row,col of each peak
def get1Dpeaks(data, 
              spacing = 10, 
              sigma = 3, 
              min_peak_size = 1, 
              max_peak_size = 50,
              include_border = False
):

  NRows, NCols = np.shape(data)

  rows = []
  cols = []
  rows_above = []
  rows_below = []
  
  tData = np.transpose(data)
  peak_size_array = np.arange(min_peak_size,max_peak_size)

  for col in np.arange(0,NCols,spacing):
    if not include_border:
      if (col == 0 or col == NCols):
        continue
    dSlice = tData[col]
    filtered = ndimage.gaussian_filter(dSlice,sigma)
    peak_rows = signal.find_peaks_cwt(filtered, peak_size_array)
    

    for row in peak_rows:
      if not include_border:
        if row == 0 or row == NRows:
          continue


      hm = filtered[row]/2.0

      nearest_below = row
      for i in np.arange(row,-1,-1):
        if (filtered[i] <= hm):
          nearest_below = i
          break


      for i in np.arange(row, len(filtered)):
        if (filtered[i] <= hm):
          nearest_above = i 
          break

      # this_width = np.abs(nearest_above - nearest_below)

      rows_above.append(nearest_above)
      rows_below.append(nearest_below)
      rows.append(row)
      cols.append(col)


  return (np.array(rows), np.array(cols)), (np.array(rows_above), np.array(rows_below))

# TODO 
# combine neighbors
# scipy.spatial.cKDTree

def genParabolas(a_inds,k_inds, aas, ks, data,pow=2): 

  NRows, NCols = np.shape(data)

  x = np.arange(0,NCols,0.1)
  ys = []

  for i in range(len(a_inds)):
    a_ind = int(a_inds[i])
    k_ind = int(k_inds[i])

    a = aas[a_ind]
    k =  ks[k_ind]

    y = a * (x)**pow + k
    y[np.extract(y, y >= NRows)] = None
    ys.append(y)

  ys = np.array(ys)
  return x,ys

def genThreeParabola(coordinates,coord_lists,data):
  a_inds, b_inds, c_inds = coordinates
  aas, bs, cs = coord_lists
  NRows, NCols = np.shape(data)
  x = np.arange(0,NCols,0.1)
  ys = []
    
  for i in range(len(a_inds)):
    a_ind = int(a_inds[i])
    b_ind = int(b_inds[i])
    c_ind = int(c_inds[i])
    
    a = aas[a_ind]
    b = bs[b_ind]
    c = cs[c_ind]
    
    y = a*x**2 + b*x + c
    y[y >= NRows] = None
    y[y < 0]      = None
    ys.append(y)
  ys = np.array(ys)
  return x,ys

def runAll():
  import flatAngles
  d = flatAngles.getFlat()
  peaks, scatterPeaks = extractOrders(d)
  HT, da = linearHoughTransform(peaks)
  a,k = findHoughPeaks(HT)
  a = a*da
  x, ys = genLines(a,k,d)

  plt.imshow(d, origin='lower')
  plt.scatter(scatterPeaks[0], scatterPeaks[1], color='red')
  for y in ys:
    plt.plot(x,y, color='y')
  plt.show()

def readData():
  filename = "sample_flat.fits"

  from astropy.io import fits
  data = fits.getdata(filename)
  return data