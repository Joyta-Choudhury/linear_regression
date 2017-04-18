# linreg-gd.py
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


def computex(xvalue):
	xvalue[0]=1.0
	return xvalue

if __name__ == "__main__":
  if len(sys.argv) !=4:
    print >> sys.stderr, "Usage: linreg-gd.py <datafile> <alpha> <# of iterations>"
    exit(-1)

  sc = SparkContext(appName="GradientDescent")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)
  #print "yxlength: ", yxlength
  
  #compute y and x values using map function
  yval = yxlines.map(lambda line: line[0]).collect()
  xval = yxlines.map(lambda line: computex(line)).collect()

  #values of y represented as matrix
  Y = np.asmatrix(yval, dtype=float)
  YT = Y.T				#transpose of Y matrix

  #values of x represented as matrix
  X = np.asmatrix(xval, dtype=float)
  XT = X.T				#transpose of X matrix
  #print Y
  #print X

  # dummy floating point array for beta to illustrate desired output format
  beta = np.zeros(yxlength, dtype=float)

  #get value of alpha and # of iterations from command line arguments
  alpha = float(sys.argv[2])
  iterations = int(sys.argv[3])

  #Calculate the shape of beta based on size of X
  beta_order = X.shape[1] 
  #beta_val = np.ones_like(X).astype(float)

  #beta initialized with values of 1
  beta_val = [1 for i in range(0,beta_order)]
  beta_iterate = []

  #converge flag to check the status if beta reached convergence
  converge_status = False
    
  #iterating the computations for # of iterations passed as an argument
  for x in range(0,iterations):
      if x == 0:
          beta_prev = beta_val
            
      #computation using gradient descent formula (beta + xtranspose.alpha.(Y-X.beta))            
      beta = np.matrix(beta_prev).T
      xbeta = np.dot(X,np.matrix(beta)) 
      yxbeta = np.subtract(YT,xbeta)
      xt_yxbeta = np.dot(XT,yxbeta)
      error = np.multiply(xt_yxbeta,alpha)
      finalbeta = np.add(beta,error)
       
      #initialized a list to append all the final beta values to beta_update
      beta_update = [] 
      for val in finalbeta.tolist():
          beta_update.append(val[0])
        
      #converge flag is set true if previous betas are equal to updated betas
      if beta_prev == beta_update:
          converge_status = True
          print('Convergence at '+ str(x) +' th position')
            
          print ("beta: ")
          for coeff in beta_update:       
              print (coeff)
          break
      #updating values of beta for each iteration
      beta_prev = beta_update
      beta_iterate = beta_update
    
  #if converge flag is false then:
  if not converge_status:
      print('No covergence till : ' + str(iterations)) 

  sc.stop()
