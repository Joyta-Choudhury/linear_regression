# linear_regression
Implementation of Linear regression using ordinary least squares:

1. Computed values of A which is multiplication of xi and xiT with the help of mapper.
2. Computed values of B which is multiplication of xi and yi with the help of mapper.
3. Used two reducers to do summation of values A and B
4. Computed B by multiplication of inverse of A and B

Implementation of Gradient Descent
1. To get X and Y values from the input file through Mapper and collect the values.
2. Initialize beta value as 1 which is of the same size as x matrix.
3. Intialize conv variable to false to check whether the convergence was achieved or not.
4. Iterate the loop until the beta value is converged.
5. Formula used to calculate beta: beta+alpha.xtranspose(y-x.beta)
6. Alpha and Iterationa are given through command line arguments.
