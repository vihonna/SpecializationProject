import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def norm_plotting(R,percent):
    # Define the mean and standard deviation for the normal distribution
    percentile = 1 - (1-percent*0.01)/2
    normal_dist = norm(0, 1)
    upper_quantile = normal_dist.ppf(percentile)  # 97.5% quantile
    std_dev = R/upper_quantile  # Standard deviation of the distribution

    x = np.linspace(-2*R, 2*R, 1000)
    normal_dist = norm(0, scale=std_dev)
    lower_quantile = normal_dist.ppf(1-percentile) 
    upper_quantile = normal_dist.ppf(percentile) 
    lower_quantile_y = normal_dist.pdf(lower_quantile)
    upper_quantile_y = normal_dist.pdf(upper_quantile)

    plt.plot(x, normal_dist.pdf(x), label=f'Percentile = {percent}%, max intensity = {round(normal_dist.pdf(0),4)}')
    plt.plot(lower_quantile, lower_quantile_y, 'ko', markersize=4)  
    plt.plot(upper_quantile, upper_quantile_y, 'ko', markersize=4)  
    plt.plot([lower_quantile, lower_quantile], [0, lower_quantile_y], color='black', linestyle='--')
    plt.plot([upper_quantile, upper_quantile], [0, upper_quantile_y], color='black', linestyle='--')
    plt.fill_between(x, 0, normal_dist.pdf(x), where=(x > lower_quantile) & (x < upper_quantile), alpha=0.3)

def uniform_plotting(R):

    x = np.linspace(-R, R, 1000)
    uniform_dist = np.ones_like(x) / (2 * R)
    plt.plot(x, uniform_dist, label=f'Radius = {R}m, max intensity = {round(1/(R*R*np.pi),4)}')
    plt.fill_between(x, 0, uniform_dist, where=(x >= -R) & (x <= R), alpha=0.3)




norm_plotting(10,99.9)
plt.xlabel('Distance relative to radius')
plt.ylabel('Density')
plt.title('Instensity distribution')
plt.legend()
plt.grid(True)
plt.show()