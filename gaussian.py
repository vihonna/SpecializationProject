import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

range = np.arange(1, 51)
var = np.zeros(len(range))
it = 0
prev_var = 0.1

def integrand(x, mean, variance):
    return norm.pdf(x, loc=mean, scale=np.sqrt(variance))

def integral_difference(variance):
    integral, _ = quad(integrand, a, b, args=(0, variance))
    return integral - constant_value

def power_law(R, a, b):
    return a * np.power(R, b)



for r in range:
    constant_value = 0.95/2
    a = 0  # Lower limit
    b = r        # Upper limit

    variance_solution = brentq(integral_difference, prev_var, 1000)

    integral, _ = quad(integrand, a, b, args=(0, variance_solution))
    var[it] = variance_solution
    it = it + 1
    prev_var = variance_solution

optimized_params, covariance = curve_fit(power_law, range, var)
optimized_a, optimized_b = optimized_params
fitted_curve = power_law(range, optimized_a, optimized_b)


plt.plot(range, fitted_curve, label=f"Optimized curve: {round(optimized_a,4)}*R^{round(optimized_b,1)}")
plt.plot(range, var, linestyle='--', label='Variance required to quarantee 0.95 light inside beam')
plt.xlabel('Range [m]')
plt.ylabel('estimated variance')
plt.legend()
plt.title('Fitted Power Law Curve')
plt.grid(True)
plt.show()