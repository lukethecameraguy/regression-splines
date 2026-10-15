# regression-splines
MTH 433 Regression Splines Work

A collection of homework exercises from MTH 433 (Applied Linear Algebra) at Miami University, solo work. Implemenets two spline-fitting methods (natural cubic spline interpolation and regression splines using a truncated power basis), both solved using a hand-written Gaussian elimination routine with partial pivoting instead of calling np.linalgsolve

## Contents
- `linalgsolve_remake.py` is a hand-made linear system solver ('luke_linalgsolve'), implementing Gaussian elimination with partial pivoting and back substitution.
- `makecoefficients.py` builds the coefficients for a natural cubic spline interpolation and solves it with `luke_linalgsolve`.
- `regression_splines_plot.py` fits a cubic regression spline to scattered data using a truncated power basis and least squares (solved with the normal equations) and then plots the fit.

##Built with
Python, Numpy, Matplotlib
