# %%
#________________________________________________________________________________________________________________________________
#                                           FILE READING
#________________________________________________________________________________________________________________________________
import pandas as pd
import re
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

filenames = ['lumaX20231025-111747.txt', 'lumaX20231025-111811.txt', 'lumaX20231025-112005.txt', 'lumaX20231025-113620.txt', 
'lumaX20231025-113709.txt', 'lumaX20231025-113753.txt','lumaX20231025-115322.txt','lumaX20231025-115359.txt', 
'lumaX20231025-122527.txt', 'lumaX20231025-122608.txt', 'lumaX20231025-123113.txt']

#--------------------------------------------------------------------------------------------------------------------------------------
#                                           Logged data for LUMAX
luma_data = []
for k in range(len(filenames)):
    txt_file_path = filenames[k]

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        # Extract every other line and split by comma
        data = [line.strip().split(',') for line in lines[::2]]
        data_time = [line.strip().split(',') for line in lines[1::2]]

    luma_data_len = len(luma_data)
    #print(data_time)
    for i in range(len(data)):
        luma_data.append([])

        date = data_time[i][0].split("-")
        luma_data[i+luma_data_len].append(int(date[-1]))
        for j in range(len(data[i])):
            value_str = (data[i][j].split(':')[1].strip())
            if value_str[-1] == '}':
                value_str = value_str[:-1] 
            value = float(value_str)
            luma_data[i+luma_data_len].append(value)            
            # match = re.search(r':\s*(\d+)\s*', data[i][j])

            # if match:
            #     number = int(match.group(1))
            #     luma_data[i+luma_data_len].append(int(match.group(1)))
            # else:
            #     print("No number found")

#--------------------------------------------------------------------------------------------------------------------------------------
#                                           Logged data for cart
#                                       The data is represented as:
#                       [0]: Time in [s]     [1]: Cartpos [m]   [2]: Velocity m/s
cart_data = []
df1 = pd.read_csv("cart1_data.txt", delimiter='\t')
df1 = df1.astype(float)
df1.iloc[:, 0] = df1.iloc[:, 0] + 40352 -25 #original40352
df1.iloc[:, 1] = 178.8 - df1.iloc[:, 1]

df2 = pd.read_csv("cart2_data.txt", delimiter='\t')
df2 = df2.astype(float)
df2.iloc[:, 0] = df2.iloc[:, 0] + 42807 #original 42807
df2.iloc[:, 1] = 178.8 - df2.iloc[:, 1]

cart_data = np.concatenate((df1.to_numpy(), df2.to_numpy()), axis=0)


#--------------------------------------------------------------------------------------------------------------------------------------
#                                     Simple function to change date to seconds
#                              Convert a clock time to seconds from 00:00:00 test day
#
def hhmmss_to_seconds(hhmmss):
    # Extract hours, minutes, and seconds
    hhmmss = str(hhmmss)
    hh = int(hhmmss[:2])
    mm = int(hhmmss[2:4])
    ss = int(hhmmss[4:])
    
    # Calculate total seconds
    return ((hh * 3600) + (mm * 60) + ss)


#--------------------------------------------------------------------------------------------------------------------------------------
#                                 Make arrays of different inputs from LumaX
#Values lumaX
time = []
freq = []
pkt_rec = []
pkt_loss = []
noise_amp = []
SNR = []
sig_strength = []
sig_amp = []
gain = []

cart_vel = []
cart_pos = []
cart_time = []

for i in range(len(luma_data)):
    time.append(hhmmss_to_seconds(luma_data[i][0]))
    #freq.append(luma_data[i][25])
    pkt_rec.append(luma_data[i][3])
    pkt_loss.append(luma_data[i][6])
    noise_amp.append(luma_data[i][12])
    SNR.append(luma_data[i][23])
    sig_strength.append(luma_data[i][15])
    sig_amp.append(luma_data[i][21])
    gain.append(luma_data[i][17])

#Values cart
for i in range(0,len(cart_data),200):
    cart_time.append(int(cart_data[i][0]))
    cart_pos.append(cart_data[i][1])
    cart_vel.append(cart_data[i][2])

#--------------------------------------------------------------------------------------------------------------------------------------
#Now i want to make an array consisting of distances and different values
#It should be of equal length to time
dist = np.zeros(len(time))
for i in range(len(time)):
    for j in range(len(cart_time)):
        if time[i] == cart_time[j]:
            dist[i] = cart_pos[j]

#--------------------------------------------------------------------------------------------------------------------------------------
#                                       Separating all experiments
exp1 = [2,86]
exp2 = [150,236]
exp3 = [315,380]

#okay so the plan is to go through all times where it has a positive vel, and pkt_rec>0
dist_useful = []
sig_amp_useful = []
noise_useful = []
SNR_useful = []
gain_useful = []
sig_str_useful = []
for i in range(len(time)):
    for j in range(len(cart_time)):
        if time[i] == cart_time[j] and cart_vel[j] < -0.05 and pkt_rec[i] > 0:
            dist_useful.append(cart_pos[j])
            sig_amp_useful.append(sig_strength[i])
            noise_useful.append(noise_amp[i])
            SNR_useful.append(SNR[i])
            gain_useful.append(gain[i])
            sig_str_useful.append(sig_strength[i])

dist_useful_2d = []
sig_amp_useful_2d = []
curr_du = []
curr_sau = []
for i in range(len(dist_useful) - 1):
    curr_du.append(dist_useful[i])
    curr_sau.append(sig_amp_useful[i])
    if dist_useful[i] > dist_useful[i + 1]:
        dist_useful_2d.append(curr_du)
        curr_du = []
        sig_amp_useful_2d.append(curr_sau)
        curr_sau = []
curr_du.append(dist_useful[-1])
dist_useful_2d.append(curr_du)
curr_sau.append(sig_amp_useful[-1])
sig_amp_useful_2d.append(curr_sau)

#--------------------------------------------------------------------------------------------------------------------------------------
#                                       Prediction functions
def pred_strength_div(x, a1,a2): 
    func = []
    for X in x:
        func.append(200*np.log10(a1*np.exp(-a2 * X)*3/(np.pi*X*X)))  
    return func

def pred_strength_gau(x, a1,a2):
    percent = 99.9
    percentile = 1 - (1-percent*0.01)/2
    normal_dist = norm(0, 1)
    z = normal_dist.ppf(percentile) 
    func = []
    for X in x:
        std = np.sqrt(3)*X/z
        func.append(200*np.log10(a1*np.exp(-a2 * X)/(np.sqrt(2*np.pi)*std)))
    return func

def pred_strength_gau2(x, a1,a2):
    percent = 95
    percentile = 1 - (1-percent*0.01)/2
    normal_dist = norm(0, 1)
    z = normal_dist.ppf(percentile) 
    func = []
    for X in x:
        std = np.sqrt(3)*X/z
        func.append(200*np.log10(a1*np.exp(-a2 * X)/(np.sqrt(2*np.pi)*std)))
    return func

def pred_strength_ch(x, a1,a2): 
    func = []
    for X in x:
        func.append(200*np.log10(a1*np.exp(-a2* X)))   
    return func

def prediction(data_x,data_y, beam):
    initial_guess = (300, 0.001)
    lower_bounds = [0, 0]   # Lower bounds for a1, a2
    upper_bounds = [10000000, 3]  # Upper bounds for a1, a2
    bounds = (lower_bounds, upper_bounds)
    if beam == 1: #divergent
        params, covariance = curve_fit(pred_strength_div, data_x, data_y, p0=initial_guess,bounds=bounds)
        a1_opt, a2_opt = params
        residuals = np.array(data_y) - np.array(pred_strength_div(data_x, a1_opt, a2_opt))
        ssr = np.sum(residuals**2)
        return pred_strength_div(np.arange(5,40,0.1), a1_opt, a2_opt), a1_opt, a2_opt, ssr        
    if beam == 2: #gaussian       
        params, covariance = curve_fit(pred_strength_gau, data_x, data_y, p0=initial_guess,bounds=bounds)
        a1_opt, a2_opt = params
        residuals = np.array(data_y) - np.array(pred_strength_gau(data_x, a1_opt, a2_opt))
        ssr = np.sum(residuals**2)
        return pred_strength_gau(np.arange(5,40,0.1), a1_opt, a2_opt), a1_opt, a2_opt, ssr   
    if beam == 3: #gaussian2       
        params, covariance = curve_fit(pred_strength_gau2, data_x, data_y, p0=initial_guess,bounds=bounds)
        a1_opt, a2_opt = params
        residuals = np.array(data_y) - np.array(pred_strength_gau2(data_x, a1_opt, a2_opt))
        ssr = np.sum(residuals**2)
        return pred_strength_gau2(np.arange(5,40,0.1), a1_opt, a2_opt), a1_opt, a2_opt, ssr   
    if beam == 4: #channel       
        params, covariance = curve_fit(pred_strength_ch, data_x, data_y, p0=initial_guess,bounds=bounds)
        a1_opt, a2_opt = params
        residuals = np.array(data_y) - np.array(pred_strength_ch(data_x, a1_opt, a2_opt))
        ssr = np.sum(residuals**2)
        return pred_strength_ch(np.arange(5,40,0.1), a1_opt, a2_opt), a1_opt, a2_opt, ssr


# %%
#________________________________________________________________________________________________________________________________
#                                           PLOTTING
#________________________________________________________________________________________________________________________________

def regression_line(x, slope, intercept):
    return slope * x + intercept

# conv_val = 10
# param_conv = np.convolve(param, np.ones(conv_val) / conv_val, mode='same')
# def plot_pred(start, end, beam, param): 
#     pred = prediction(start, end, beam, param)
#     plt.plot(dist[start:end], pred[0], label=f"Beamtype {beam} with a1: {pred[1]}, a2: {pred[2]}")   

# def plot_scatter(start, end, param):
#     plt.scatter(dist[start:end], param[start:end])

# startval = exp2[0]
# endval = exp2[1]+1
# param = sig_amp
legend_text = ["10 Mhz, normal light","10MHz, normal light","1Mhz, normal light", "1MHz, minimum light", "1MHz, minimum light"]
for i in range(len(dist_useful_2d)):
    plot_array_x = np.array(dist_useful_2d[i])[~np.isnan(np.array(dist_useful_2d[i]))]
    plot_array_y = np.array(sig_amp_useful_2d[i])[~np.isnan(np.array(sig_amp_useful_2d[i]))]
    plt.scatter(plot_array_x,plot_array_y, label=f"{legend_text[i]}")
    y = prediction(plot_array_x,plot_array_y, 2)
    plt.plot(np.arange(5,40,0.1), y[0], label=f"Fitted curve, alpha = {round(y[2],3)}")

#               PLOTTING OF 1 AND 10 MHz
#plt.scatter(dist_useful,sig_amp_useful, label=f"Data points ")
# plt.scatter(dist_useful[40:63],noise_useful[40:63], label=f"Data points 1MHz, ambient noise")
# slope10, intercept10 = np.polyfit(dist_useful[:40],noise_useful[:40], 1)
# slope1, intercept1 = np.polyfit(dist_useful[40:63],noise_useful[40:63], 1)
# m1 = np.mean(noise_useful[:40])
# m2 = np.mean(noise_useful[40:63])
# x_arr = np.arange(0,30,0.1)
# func10 = []
# func1 = []
# for x in x_arr: 
#     func10.append(regression_line(x,slope10,intercept10))
#     func1.append(regression_line(x,slope1,intercept1))
# plt.plot(x_arr,func10, label=f"Regression line, bandwidth 10MHz, mean = {round(m1,2)}")
# plt.plot(x_arr,func1, label=f"Regression line, bandwidth 10MHz, mean = {round(m2,2)}")


# y_gaus99_10 = prediction(dist_useful[:40],SNR_useful[:40], 2)
# y_gaus99_1 = prediction(dist_useful[40:68],SNR_useful[40:68], 2)
# plt.plot(np.arange(5,50,0.1), y_gaus99_10[0], label=f"Bandwidth 10MHz, I0 = {round(y_gaus99_10[1])}, alpha = {round(y_gaus99_10[2],3)}, SSR = {round(y_gaus99_10[3],3)}")
# plt.plot(np.arange(5,50,0.1), y_gaus99_1[0], label=f"Bandwidth 1MHz, I0 = {round(y_gaus99_1[1])}, alpha = {round(y_gaus99_1[2],3)}, SSR = {round(y_gaus99_1[3],3)}")



#plt.scatter(dist_useful,noise_useful)
# plot_pred(startval,endval,2,sig_amp)
# #plot_pred(startval,endval,1,sig_amp)
# plot_scatter(startval,endval,sig_amp)
# plot_scatter(startval,endval,noise_amp)
#plt.scatter(dist_useful,sig_amp_useful)
plt.title("Gaussian beam at 0.999 confidence interval")
plt.xlabel("Distance [m]")
plt.ylabel("Signal amplitude []")
plt.legend()
plt.grid()
plt.show()

# %%
