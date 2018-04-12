import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.ndimage.filters as filters
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
def extractOrders(externalData, medianFilterShape=(2,1), verbose=True):
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
  data = filters.median_filter(data, size=windowShape)


  if verbose:
    print("Finding Peaks")
  #find peaks column wise
  peaks = get1Dpeaks(data)

  # Hough Transform:
  # y = ax**2 + bx + c

  aMin = -2e-5
  aMax = 2e-4
  aLen = 60
  aList = np.linspace(aMin, aMax, aLen)

  bMin = 0.0
  bMax = 0.2
  bLen = 60
  bList = np.linspace(bMin, bMax, bLen)

  cMin = -10
  cMax = np.shape(data)[0]
  cList = np.arange(cMin,cMax+1,1)


  if verbose:
    print("Performing Hough Transform")
  HT = hough.parabolicHT(peaks,(aList,bList,cList),verbose=True)

  if verbose:
    print("Seaching for Peaks")
  found_as, found_bs, found_cs, ht_maxima, ht_maxFilt = findThreePeaks(HT)


  if verbose:
    print("Plotting")
  x,ys = genThreeParabola(found_as,found_bs,found_cs,aList,bList,cList,data)

  plt.imshow(data)
  plt.scatter(peaks[1],peaks[0])

  for y in ys:
    plt.plot(x,y,color='k')
  plt.show()

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
  tData = np.transpose(data)
  peak_size_array = np.arange(min_peak_size,max_peak_size)

  for col in np.arange(0,NCols,spacing):
    if not include_border:
      if (col == 0 or col == NCols):
        continue
    dSlice = tData[col]
    filtered = filters.gaussian_filter(dSlice,sigma)
    peak_rows = signal.find_peaks_cwt(filtered, peak_size_array)
    
    for row in peak_rows:
      if not include_border:
        if row == 0 or row == NRows:
          continue
      rows.append(row)
      cols.append(col)

  return np.array(rows), np.array(cols)



def findThreePeaks(HT):
  #local maximum filter
  nieghborhood_size = 40
  half_max = np.max(HT)/2.0


  CHT = clean(HT,half_max)
  CHT_max = filters.maximum_filter(CHT, nieghborhood_size)
  maxima = (CHT == CHT_max)

  #delete zeros
  maxima[CHT_max == 0] = 0



  labeled, numObjects = ndimage.label(maxima) #, structure = np.ones([3,3], bool))
  slices = ndimage.find_objects(labeled)
    
  aaa,bbb,ccc = [], [], []
  for da,db,dc in slices:
      a_center = (da.start + da.stop -1)/2
      aaa.append(a_center)

      b_center = (db.start + db.stop -1)/2
      bbb.append(b_center)

      c_center = (dc.start + dc.stop - 1)/2    
      ccc.append(c_center)

  return aaa, bbb, ccc, maxima, CHT_max

#sets everything below t to zero in arr
def clean(arr, t):
  cp = []
  cp.extend(arr)
  cp = np.array(cp)

  cp[cp<t] = 0
  return cp

# TODO 
# combine neighbors
# scipy.spatial.cKDTree

def findHoughPeaks(HT):
  #local maximum filter
  nieghborhood_size = 40
  half_max = np.max(HT)/2.0


  CHT = clean(HT,half_max)
  CHT_max = filters.maximum_filter(CHT, nieghborhood_size)
  maxima = (CHT == CHT_max)

  #delete zeros
  maxima[CHT_max == 0] = 0



  labeled, numObjects = ndimage.label(maxima) #, structure = np.ones([3,3], bool))
  slices = ndimage.find_objects(labeled)
  a,k = [], []
  for dk, da in slices:
    a_center = (da.start + da.stop -1)/2
    a.append(a_center)

    k_center = (dk.start + dk.stop - 1)/2    
    k.append(k_center)


  return a, k, maxima

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

def genThreeParabola(a_inds,b_inds,c_inds,aas,bs,cs,data):
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
    # y[np.extract(y,y > NRows)] = None
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