# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

grid = 500

x_arr = np.linspace(0.1, 50, grid)
theta = np.linspace(0, np.pi/2, grid)

#Params
W = 1
N0 = 1
alpha = 0.018
quantile = 0.9999
L0 = 80
SNR_min = 0.1

def SNR_g(th, x, W, N0, alpha, quantile, L0):
    a = 1
    L = L0*np.exp(-alpha*x/np.cos(th))
    zed = norm.ppf(quantile)
    std_dev = x*np.tan(np.pi/3)/zed
    BP = 1/(np.sqrt(2*np.pi)*std_dev) * np.exp(-0.5*(x*np.tan(th)/std_dev)**2)
    SNR = a*(L*BP)**2/(4*W*N0)
    return SNR

def SNR_d(th, x, W, N0, alpha, L0):
    a = 1
    L = L0*np.exp(-alpha*x/np.cos(th))
    BP = 1/(np.pi*(x*np.sin(np.pi/3))**2)
    return a*(L*BP)/(W*N0)

for k in range(1,11):
    r_g = np.zeros_like(x_arr)
    for i in range(len(x_arr)):
        for th in theta:
            SNR_g_temp = SNR_g(th, x_arr[i], k, N0, alpha, quantile, L0)
            if SNR_g_temp > SNR_min:
                r_g[i] = x_arr[i]*np.tan(th)
    plt.plot(r_g, x_arr, label=f"Bandwidth = {k}MHz")

# Show the plot
plt.title("Max range, given gaussian beam at 0.9999 confidence interval")
plt.xlabel("Radius [m]")
plt.ylabel("Distance in z-direction [m]")
plt.legend()
plt.grid(True)
plt.show()
# %%
