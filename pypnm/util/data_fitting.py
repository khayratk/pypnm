from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt


def residuals(p, function, y, x):
    """
    Evaluate residuals between data and fitting function
    Args:
       p: Parameters of fitting function.
       function: Fitting function.
       y: Data to be fitted to.
       x: Domain of fitting function.
    """
    err = y - function(x, p)
    return err


def fit_params(fit_func, n_params, y, x, p0=0):
    """ 
    Fits the parameters of a function of the form function(x,p) where x is
    the domain of the fitting function and p is an array containing its
    parameters.
    Args:
       fit_func: The parametric function to be fitted to the data.
       n_params: Number of parameters of the function.
       y: Data to be fitted to.
       x: Domain of the fitting function.
    """

    if np.all(np.array(p0) == 0):
        p0 = [np.sqrt(2)]*n_params

    plsq = leastsq(residuals, p0, args=(fit_func, y, x))
    p_fitted = plsq[0]

    return p_fitted


if __name__ == "__main__":
    """ Example of usage """

    def myfunc(x, p):
        return p[0] * x + p[1] * x ** 2

    def myfunc2(x, p):
        return p[1] + x ** p[0]

    def myfunc3(x, p):
        return np.tanh(p[0] * x ** p[2]) * p[1]

    x = np.array(range(1.0, 20)) * 0.1
    y = np.tanh(x)

    params_myfunc = fit_params(myfunc, 2, y, x)
    params_myfunc2 = fit_params(myfunc2, 2, y, x)
    params_myfunc3 = fit_params(myfunc3, 3, y, x, p0=[2, 4., 5.])

    print params_myfunc
    print params_myfunc2
    print params_myfunc3

    plt.plot(x, y, '.')
    plt.plot(x, myfunc(x, params_myfunc), 'r')
    plt.plot(x, myfunc2(x, params_myfunc2), 'b')
    plt.plot(x, myfunc3(x, params_myfunc3), 'black')

    plt.show()
