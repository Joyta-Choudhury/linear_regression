# linreg.py
# jchoudh1@uncc.edu
# Joyta Choudhury
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.


import sys
import numpy as np

from pyspark import SparkContext
from numpy.linalg import inv

#function to compute values of a with the help of x values
def computea(xvalues):
	xmatrix = np.matrix(xvalues,dtype=float)	#forms a row matrix
	xtranspose = np.insert(xmatrix, 0, 1, axis=1)	#insert 1 at 0 index
	x = xtranspose.T				#transpose of a matrix which will be x matrix
	mula = np.dot(x,xtranspose)			#multiply x and x transpose to get value of a
	return mula
	
#function to compute values of b with the help of x values and y value
def computeb(yvalue,xvalues):
	xmatrix = np.matrix(xvalues,dtype=float)	#forms a row matrix
	xtranspose = np.insert(xmatrix, 0, 1, axis=1)	#insert 1 at 0 index
	x = xtranspose.T				#transpose of a matrix which will be x matrix
	ymatrix = np.matrix(yvalue,dtype=float)		#forms a matrix for y
	mulb = np.dot(x,ymatrix)			#multiply x and y matrxix to get value of b
	return mulb

if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)
  #print "yxlength: ", yxlength

  # dummy floating point array for beta to illustrate desired output format
  beta = np.zeros(yxlength, dtype=float)
  
  #Using map and reduce to get values of a and b
  valuesa = yxlines.map(lambda line: computea(line[1:])).reduce(lambda x,y: np.add(x,y))
  valuesb = yxlines.map(lambda line: computeb(line[0],line[1:])).reduce(lambda x,y: np.add(x,y))
 
  #print valuesa
  #print valuesb

  #taking inverse of a
  inversea = inv(np.matrix(valuesa))
  
  #computing values of beta by multiplying inverse of a with b
  beta = np.dot(inversea,valuesb)
  
  #to save as an output file
  #np.savetxt('yxlin1-s.out',beta)

  # print the linear regression coefficients in desired output format
  print "beta: "
  for coeff in beta:
      print coeff

  sc.stop()
