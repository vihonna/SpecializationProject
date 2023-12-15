# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import norm

#const 
alp = 1
N = 1
c2 = 1
c1 = 1
a = 1
z = norm.ppf(0.999)
W = 1
eps = 1

#vars 
p = np.linspace(0.01,1,100)
x = np.linspace(0.1,10,100)

p, x = np.meshgrid(p, x)
r = eps*np.sqrt((1 + x*c2)**2 + (1 + x*c1)**2)
std = np.sqrt(3)*x/z
snr = 1/(N*W)*p * np.exp(-alp*x) * 1/(np.sqrt(2*np.pi)*std) * np.exp(-0.5*(r*r)/(std*std))
#snr = 1/(N*W)*p * np.exp(-alp*r) * 1/(np.pi*x*x) 

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(p, r, snr, cmap='viridis')

# Add labels
ax.set_xlabel('power')
ax.set_ylabel('Distance away in z-direction')
ax.set_zlabel('SNR')

# Show plot
plt.show()
# %%
