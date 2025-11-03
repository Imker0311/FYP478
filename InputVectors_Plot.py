# %%

import numpy as np
import matplotlib.pyplot as plt
import h5py

filename = 'InputVectors_Test.h5'

def load_vector(name):
    with h5py.File(filename, 'r') as f:
        data = f[name][:]
        print(f"Loaded dataset '{name}' with shape {data.shape}")
        return data

t = load_vector('t')/3600

F1_vec = load_vector('F1')
F2_vec = load_vector('F2')
F3_vec = load_vector('F3')
F4_vec = load_vector('F4')
F5_vec = load_vector('F5')
F6_vec = load_vector('F6')
F7_vec = load_vector('F7')
F8_vec = load_vector('F8')
F9_vec = load_vector('F9')

F1_plt = load_vector('F1.plt')
F2_plt = load_vector('F2.plt')
F3_plt = load_vector('F3.plt')
F4_plt = load_vector('F4.plt')
F5_plt = load_vector('F5.plt')
F6_plt = load_vector('F6.plt')
F7_plt = load_vector('F7.plt')
F8_plt = load_vector('F8.plt')
F9_plt = load_vector('F9.plt')

plt.figure(figsize=(10, 25))
plt.subplot(10, 1, 1)
plt.scatter(t,F1_plt*1,s=10)
plt.scatter(t,F2_plt*2,s=10)
plt.scatter(t,F3_plt*3,s=10)
plt.scatter(t,F4_plt*4,s=10)
plt.scatter(t,F5_plt*5,s=10)
plt.scatter(t,F6_plt*6,s=10)
plt.scatter(t,F7_plt*7,s=10)
plt.scatter(t,F8_plt*8,s=10)
plt.scatter(t,F9_plt*9,s=10)
plt.ylim(0.5, 9.5)
plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9], ['(F1)', '(F2)', '(F3)', '(F4)', '(F5)', '(F6)', '(F7)', '(F8)', '(F9)'])
plt.ticklabel_format(style='plain', axis='x')
plt.xticks(np.arange(0, 850, 50))
plt.xlabel('Time [h]')
plt.ylabel('Faults')
plt.grid()==True

plt.figure(figsize=(10, 16))
plt.subplot(10, 1, 2)
plt.plot(t, F1_vec, label='E/R')
plt.ylabel('E/R')

plt.subplot(10, 1, 3)
plt.plot(t, F2_vec, label='U_Ac')
plt.ylabel('U_Ac')

plt.subplot(10, 1, 4)
plt.plot(t, F3_vec, label='T Bias')
plt.ylabel('T Bias')

plt.subplot(10, 1, 5)
plt.plot(t, F4_vec, label='T_F')
plt.ylabel('T_F')

plt.subplot(10, 1, 6)
plt.plot(t, F5_vec, label='C_F')
plt.ylabel('C_F')

plt.subplot(10, 1, 7)
plt.plot(t, F6_vec, label='T_CF')
plt.ylabel('T_CF')

plt.subplot(10, 1, 8)
plt.plot(t, F7_vec, label='Q_F')
plt.ylabel('Q_F')

plt.subplot(10, 1, 9)
plt.plot(t, F8_vec, label='dP')
plt.ylabel('dP')

plt.subplot(10, 1, 10)
plt.plot(t, F9_vec, label='dPc')
plt.ylabel('dPc')


