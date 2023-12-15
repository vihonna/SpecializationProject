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
from collections import defaultdict
from scipy.stats import gamma

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import expon

filenames = ['chann_json_DRONE.txt']
filenames2 = ['chann_json_BENTHIC.txt']

#--------------------------------------------------------------------------------------------------------------------------------------
#                                           Logged data for LUMAX
luma_data = []
for k in range(len(filenames)):
    txt_file_path = filenames[k]

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        # Extract every other line and split by comma
        data = [line.strip().split(',') for line in lines[1::2]]
        data_time = [line.strip().split(',') for line in lines[::2]]

    luma_data_len = len(luma_data)
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

luma_data2 = []
for k in range(len(filenames2)):
    txt_file_path = filenames2[k]

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        # Extract every other line and split by comma
        data = [line.strip().split(',') for line in lines[1::2]]
        data_time = [line.strip().split(',') for line in lines[::2]]

    luma_data_len = len(luma_data2)
    for i in range(len(data)):
        luma_data2.append([])

        date = data_time[i][0].split("-")
        luma_data2[i+luma_data_len].append(int(date[-1]))
        for j in range(len(data[i])):
            value_str = (data[i][j].split(':')[1].strip())
            if value_str[-1] == '}':
                value_str = value_str[:-1] 
            value = float(value_str)                 
            luma_data2[i+luma_data_len].append(value)

#--------------------------------------------------------------------------------------------------------------------------------------
#                                     Simple function to change date to seconds
#                              Convert a clock time to seconds from 00:00:00 test day
#
def hhmmss_to_seconds(hhmmss):
    hhmmss = str(hhmmss)
    hh = int(hhmmss[:2])
    mm = int(hhmmss[2:4])
    ss = int(hhmmss[4:])
    return ((hh * 3600) + (mm * 60) + ss)
#--------------------------------------------------------------------------------------------------------------------------------------
#                                           Logged data for iperf
iperf_data = []
current_array = []
with open('iperf_DRONE_copy.txt', 'r') as file:
    for line in file:
        line = line.strip() 
        if len(line) == 7:
            if current_array:
                iperf_data.append(current_array)
            current_array = []
            current_array.append(line[1:3] + line[4:6] + "00")
        else:
            elements = line.split()  
            bw_iperf = float(elements[6])
            if bw_iperf > 10 : bw_iperf = bw_iperf/1000
            current_array.append(bw_iperf)  

#--------------------------------------------------------------------------------------------------------------------------------------
#                                 Make arrays of different inputs from LumaX
#Values lumaX
time = []
bw = []
pkt_rec = []
pkt_loss = []
SNR = []
crc = []
speed = []
SNR_int=[]
bin_pkt = []

for i in range(len(luma_data)):
    time.append(hhmmss_to_seconds(luma_data[i][0]))
    bw.append(luma_data[i][4]) 
    pkt_rec.append(luma_data[i][6])
    pkt_loss.append(luma_data[i][7])
    SNR.append(luma_data[i][22])
    crc.append(luma_data[i][5])
    speed.append(luma_data[i][25])

    #Indirect datasets
    SNR_int.append(int(luma_data[i][22]))
    if luma_data[i][6] < 0.5: bin_pkt.append(0)
    if luma_data[i][6] > 0.5: bin_pkt.append(6)

#Values lumaX for BENTHIC
time2 = []
bw2 = []
pkt_rec2 = []
pkt_loss2 = []
SNR2 = []


for i in range(len(luma_data)):
    time2.append(hhmmss_to_seconds(luma_data[i][0]))
    bw2.append(luma_data[i][4]) 
    pkt_rec2.append(luma_data[i][4])
    pkt_loss2.append(luma_data[i][6])
    SNR2.append(10*luma_data[i][22])




#--------------------------------------------------------------------------------------------------------------------------------------
#                                           Data syncronizing 

#iperf_data_1d = []
#shifts = [-25,27,0,18,0,-24,0,139,139,17]
iperf_times = []
#iperf_data_1ds = []
for i in range(len(iperf_data)):
    iperf_times.append([])
    for d in range(120):
        iperf_times[i].append([])
        for j in range(1,len(iperf_data[i])):
            if iperf_data[i][j] == np.nan: break
            iperf_times[i][d].append(hhmmss_to_seconds(iperf_data[i][0]) + j - 1 + d-60)
            #iperf_data_1ds[i][d].append(iperf_data[i][j])

iperf_shorts = []
iperf_SNRs = []
ideal_corrs = []
iperf_short_new = []
iperf_SNR_new = []
iperf_times_2d = []
iperf_times_3d = []
for i in range(len(iperf_times)):
    iperf_shorts.append([])
    iperf_SNRs.append([])
    iperf_times_3d.append([])
    corrs = []
    for d in range(len(iperf_times[i])):
        iperf_shorts[i].append([])
        iperf_SNRs[i].append([])
        iperf_times_3d[i].append([])
        for k in range(len(iperf_times[i][d])):
            for j in range(1,len(time)):
                if iperf_times[i][d][k] == time[j]: 
                    iperf_shorts[i][d].append(iperf_data[i][k+1])
                    iperf_SNRs[i][d].append(SNR[j])
                    iperf_times_3d[i][d].append([iperf_times[i][d][k]])
        corr = spearmanr(iperf_SNRs[i][d], iperf_shorts[i][d])
        corrs.append(corr[0])
    ideal_corrs.append(np.argmax(corrs))
    iperf_short_new.append(iperf_shorts[i][ideal_corrs[i]])
    iperf_SNR_new.append(iperf_SNRs[i][ideal_corrs[i]])
    iperf_times_2d.append(iperf_times_3d[i][ideal_corrs[i]])

iperf_times_1d = [element for sublist in iperf_times_2d for element in sublist]
iperf_SNR_ideal = [element for sublist in iperf_SNR_new for element in sublist]
iperf_short_ideal = [element for sublist in iperf_short_new for element in sublist]

iperf_time = []
for i in range(len(iperf_data)):
    for j in range(1,len(iperf_data[i])):
        if iperf_data[i][j] == np.nan: break
        iperf_time.append(hhmmss_to_seconds(iperf_data[i][0]) + j - 1)
iperf_data_1d = [element for subarray in iperf_data for element in subarray[1:]]

iperf_short = []
iperf_SNR = []
for i in range(len(iperf_time)):
    for j in range(1,len(time)):
        if iperf_time[i] == time[j]:
            iperf_short.append(iperf_data_1d[i])
            iperf_SNR.append(SNR[j])

#--------------------------------------------------------------------------------------------------------------------------------------
#                                       Signal disruption
sig_dis = []
for i in range(len(pkt_rec)):
    if (pkt_rec[i] == 0) and (pkt_rec[i-1] != 0):
        disconnect = SNR[i]
        it = 1
        while disconnect == 0:
            disconnect = SNR[i-it]
            it = it + 1
        sig_dis.append(disconnect)

def hist():
    weight = 1/len(sig_dis)
    bin_size = 0.05
    max = 1.5
    bins = np.zeros(int(max/bin_size)) #max can be decided later by max(sig_dis)
    for i in range(len(sig_dis)):
        if sig_dis[i] > 1.49:
            bins[-1] = bins[-1] + weight
        else: 
            bins[int(sig_dis[i]/bin_size)] = bins[int(sig_dis[i]/bin_size)] + weight
    x_bins = np.arange(0, max, bin_size)
    return x_bins, bins


#--------------------------------------------------------------------------------------------------------------------------------------
#                                       Data processing
def hist_with_max(param):
    weight = 1/len(sig_dis)
    bin_size = 0.05
    max = 2
    bins = np.zeros(int(max/bin_size)) #max can be decided later by max(sig_dis)
    for i in range(len(sig_dis)):
        if sig_dis[i] > 1.99:
            bins[-1] = bins[-1] + weight
        else: 
            bins[int(sig_dis[i]/bin_size)] = bins[int(sig_dis[i]/bin_size)] + weight
    x_bins = np.arange(0, max, bin_size)
    return x_bins, bins

pkt_by_SNR = defaultdict(list)
for i, xi in enumerate(SNR_int):
    pkt_by_SNR[xi].append(pkt_rec[i])

mean_pkt = [sum(pkt_by_SNR[xi]) / len(pkt_by_SNR[xi]) for xi in sorted(pkt_by_SNR.keys())]
x_positions = sorted(pkt_by_SNR.keys())


# %%
#________________________________________________________________________________________________________________________________
#                                           PLOTTING
#________________________________________________________________________________________________________________________________
from scipy.stats import expon



#                                           DELAY PLOTTING
#________________________________________________________________________________________________________________________________

mean_corr1 = spearmanr(iperf_SNR,iperf_short)
mean_corr = spearmanr(iperf_SNR_ideal, iperf_short_ideal)
SNR_rounded = np.round(np.array(iperf_SNR_ideal),decimals=0)
rounded_dict = {}
for i, val in enumerate(SNR_rounded):
    if val not in rounded_dict:
        rounded_dict[val] = []
    rounded_dict[val].append(iperf_short_ideal[i])
means_dict = {key: np.mean(values) for key, values in rounded_dict.items()}
sorted_means_dict = {k: means_dict[k] for k in sorted(means_dict)}
SNR_int = np.zeros(int(max(iperf_SNR_ideal)))
SNR_int_x = np.arange(0,int(max(iperf_SNR_ideal)))
it = 0
for i in range(int(max(iperf_SNR_ideal))):
    for j in range(len(iperf_SNR_ideal)):
        if int(iperf_SNR_ideal[j] == i):
            SNR_int[i] = SNR_int[i] + iperf_short_ideal[j]
            it = it + 1
    if it != 0 : SNR_int[i] = SNR_int[i]/it
dg = list(sorted_means_dict)
beta = np.var(dg)/np.mean(dg)
alpha = np.mean(dg)/beta
x = np.linspace(0, 30, 1000)

#------------------------------------------------------------------------------------------------------------------
#                                   GAMMA
# plt.scatter(SNR_int_x,SNR_int)
#plt.plot(x, pdf, 'r-', label='Gamma PDF*100')
#plt.plot(x, pdf2, 'b-', label='Normal PDF')
# bins = hist()
# plt.scatter(bins[0], bins[1])
#plt.plot(time, SNR, label="SNR*0.1")
# plt.scatter(iperf_time, iperf_data_1d, label="Original", color="black", s=5)
# plt.scatter(iperf_times_1d, iperf_short_ideal, label="Time shifted", color="grey", s=5)
# plt.plot(iperf_SNR_ideal, y1)
#plt.scatter(iperf_SNR_ideal, iperf_short_ideal, label=f"Delay adjustments, correlation = {round(mean_corr[0],2)}")
x,y = hist()
# plt.bar(x, y, width=0.02, label="Histogram of SNR´s when the transmission is lost")
tmp=0
sum50=0
while(sum50<0.5):
    sum50 = sum50+y[tmp]
    tmp = tmp + 1
# beta = np.var(sig_dis)/np.mean(sig_dis)
plt.scatter(list(sorted_means_dict.keys()), list(sorted_means_dict.values()), label=f"Time shifted data")
# plt.scatter(iperf_SNR, iperf_short, label=f"No delay adjustment, correlation = {round(mean_corr1[0],2)}")
#exp_pdf = gamma.cdf(np.linspace(0, 25, 1000),a=np.mean(sig_dis)/beta, scale=beta)
#cdf_log = np.log10(exp_pdf)
#exp_pdf = gamma.cdf(np.linspace(0, 100, 1000),a=np.mean(sig_dis)/beta, scale=beta)
#plt.plot(np.linspace(0.02, 1.5, 1000), exp_pdf, color='black', label='Cumulative gamma distribution')
plt.axvline(x=x[tmp], color='grey', linestyle='--', label=f'Mean SNR_cutoff: {x[tmp]}')



plt.xlabel("SNR")
#plt.ylabel("Density")
plt.ylabel("Throughput [MBit/s]")
#plt.scatter(iperf_SNR_s, iperf_short_s, label="shifted")
#plt.scatter(SNR, bw, label="bw")
#plt.plot(time2, SNR2, label="bw2")
plt.legend()
plt.show()
# %%