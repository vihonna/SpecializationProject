import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def norm_plotting(R,percent):
    percentile = 1 - (1-percent*0.01)/2
    normal_dist = norm(0, 1)
    upper_quantile = normal_dist.ppf(percentile)  # 97.5% quantile
    std_dev = R/upper_quantile  # Standard deviation of the distribution
    angle = np.linspace(0,65,6500)
    x = np.tan(np.deg2rad(angle))*R
    normal_dist = norm(0, scale=std_dev)
    y = normal_dist.pdf(x)
    plt.plot(angle, y, label=f'Normal distribution, {percent}%, max intensity = {round(normal_dist.pdf(0),4)}')


def uniform_plotting(R,max_angle):
    angle = np.linspace(0,65,6500)
    p0 = 1/(R*R*np.pi)
    y=[]
    for ang in angle:
        if ang<max_angle: y.append(p0)
        else: y.append(0)
    plt.plot(angle, y, label=f'Uniform distribution, Imax = {round(1/(1*1*np.pi),4)}')


#uniform_plotting(10,60)
norm_plotting(10,99)

plt.xlabel('Misalignment [deg]')
plt.ylabel('Intensity []')
plt.title('Instensity for R=10m')
plt.legend()
plt.grid(True)
plt.show()