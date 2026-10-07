import json 
import os 
import numpy as np 
import matplotlib.pyplot as plt 

t_r_sigma = [] 
t_sigma_r = []

for i in range(10):
    json_file = open('./sample_ini_conds/FV_replica_' + str(i) + '/ini_fv_' + str(i) + '_checkpoint.txt', 'r')
    checkpoint_data = json.load(json_file)
    json_file.close() 
    t_r_sigma = t_r_sigma + checkpoint_data["t_r_sigma"][0]
    t_sigma_r = t_sigma_r + checkpoint_data["t_sigma_r"][0]

t_r_sigma = np.array(t_r_sigma) 
t_sigma_r = np.array(t_sigma_r) 

t_loop = (t_r_sigma + t_sigma_r) * 1.0  #in fempto seconds 
np.savetxt('t_loop.txt', t_loop)

plt.figure()
plt.hist(t_loop, 200)
plt.show()
plt.savefig('hist_t_loop.png', dpi=160) 