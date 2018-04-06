import numpy as np
import scipy.ndimage as ndimage
import scipy.signal  as signal
import math


# data: data to be transformed: lists of rows,cols of points
# func: function to look for in form func(x,y,params) = y
# def transform(data, func, paramRanges):


#param arrays should be sorted
def linearHT(data,paramArrays):
  # y = mx+b

  mList = paramArrays[0]
  bList = paramArrays[1]
  delta = mList[1] - mList[0]
  lower_limit = mList[0]  - delta
  upper_limit = mList[-1] + delta

  # Accumulator
  A = np.zeros([len(mList),len(bList)])

  rows,cols = data

  logRow = 0
  for row,col in zip(rows,cols):
    
    if row>logRow:
      print('row: '+str(logRow))
      logRow += 100

    #parameter of singular values
    #have option to default to try/catch
    if col == 0:
      continue

    found_ms = (row - bList) / (col)
    to_increment = [(np.abs(mList-m).argmin(),b_index) for b_index,m in enumerate(found_ms) if m>=lower_limit and m<=upper_limit]
    for nearest_m,b_index in to_increment:
      A[nearest_m,b_index] = A[nearest_m,b_index]+1
    
  return A

def parabolicHT(data, paramArrays,verbose=False):
  # y = a*x^2 + bx + c

  aList = paramArrays[0]
  bList = paramArrays[1]
  cList = paramArrays[2]

  delta = aList[1] - aList[0]

  lower_limit = aList[0]  - delta
  upper_limit = aList[-1] + delta


  # A = np.zeros([len(ks), bLen, aLen ])

  A = np.zeros([len(aList),len(bList),len(cList)])

  rows,cols = data

  
  logCountBase = 0.1*len(rows)
  logCount = 0
  count = 0
  for row,col in zip(rows,cols):
    count+=1
    if verbose and count>logCount:
      print('Percent: '+str(logCount/len(rows)*100))
      logCount += logCountBase

    #parameter of singular values
    #have option to default to try/catch
    if col == 0:
      continue

    for c_index,c in enumerate(cList):
      found_as = (row - c)/(col**2) - bList/col
      to_increment = [(np.abs(aList-a).argmin(),b_index) for b_index,a in enumerate(found_as) if a>=lower_limit and a<=upper_limit]
      for nearest_a,b_index in to_increment:
        A[nearest_a,b_index,c_index] = A[nearest_a,b_index,c_index]+1
    
  return A
