# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_vector_appended(filename, dataset_name):

    with h5py.File(filename, 'r') as f:
        data = f[dataset_name][:]
        print(f"Loaded dataset '{dataset_name}' with shapes {data.shape}")
    return data

file = 'CSTR_Simulation_Train.h5'

Time = load_vector_appended(file, 'Time')
C_A = load_vector_appended(file, 'C_A')  #1
T = load_vector_appended(file, 'T')  #2
T_C = load_vector_appended(file, 'T_C')  #3
h = load_vector_appended(file, 'h')  #4
Q_SP = load_vector_appended(file, 'Q_SP')  #5
Qc_SP = load_vector_appended(file, 'Qc_SP')  #6
l = load_vector_appended(file, 'l')  #7
l_c = load_vector_appended(file, 'l_c')  #8
Q = load_vector_appended(file, 'Q_vec')  #9
Q_c = load_vector_appended(file, 'Q_c_vec')  #10

file = 'InputVectors_Train.h5'

E_R = load_vector_appended(file, 'F1')[1:]
U_Ac = load_vector_appended(file, 'F2')[1:]
T_bias = load_vector_appended(file, 'F3')[1:]
T_F = load_vector_appended(file, 'F4')[1:]   #14
C_AF = load_vector_appended(file, 'F5')[1:]  #11
T_CF = load_vector_appended(file, 'F6')[1:]  #12
Q_F = load_vector_appended(file, 'F7')[1:]  #13
dP = load_vector_appended(file, 'F8')[1:]
dPc = load_vector_appended(file, 'F9')[1:]


plot_var = E_R
plot_var_name = r"E/R (K)"

#averages
C_A_avg = np.mean(C_A)
T_avg = np.mean(T)
T_C_avg = np.mean(T_C)
h_avg = np.mean(h)
Q_SP_avg = np.mean(Q_SP)
Qc_SP_avg = np.mean(Qc_SP)
l_avg = np.mean(l)
l_c_avg = np.mean(l_c)
Q_avg = np.mean(Q)
Q_c_avg = np.mean(Q_c)


# Print averages
print(f"C_A_avg: {C_A_avg:.4f}")
print(f"T_avg: {T_avg:.4f}")
print(f"T_C_avg: {T_C_avg:.4f}")
print(f"h_avg: {h_avg:.4f}")
print(f"Q_SP_avg: {Q_SP_avg:.4f}")
print(f"Qc_SP_avg: {Qc_SP_avg:.4f}")
print(f"l_avg: {l_avg:.4f}")
print(f"l_c_avg: {l_c_avg:.4f}")
print(f"Q_avg: {Q_avg:.4f}")
print(f"Q_c_avg: {Q_c_avg:.4f}")


# %%
# Plotting

plt.figure(figsize=(8, 4))
Time = Time / 3600

plt.plot(Time, plot_var)
plt.xlabel('Time (h)')
plt.ylabel(plot_var_name)

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(Time, C_A)
plt.xlabel('Time (h)')
plt.ylabel('$C_{A}$ (mol/L)')

plt.subplot(3, 2, 2)
plt.plot(Time, h)
plt.xlabel('Time (h)')
plt.ylabel('H (m)')

plt.subplot(3, 2, 3)
plt.plot(Time, T)
plt.xlabel('Time (h)')
plt.ylabel('T (K)')

plt.subplot(3, 2, 4)
plt.plot(Time, T_C)
plt.xlabel('Time (h)')
plt.ylabel('$T_{C}$ (K)')

plt.subplot(3, 2, 5)
plt.plot(Time, Q)
plt.xlabel('Time (h)')
plt.ylabel('Q (L/s)')

plt.subplot(3, 2, 6)
plt.plot(Time, Q_c)
plt.xlabel('Time (h)')
plt.ylabel('$Q_{c}$ (L/s)')


# %%
xlim=[0, 820]

# Plot numbered vectors above
plt.figure(figsize=(14, 12))
plt.subplot(7, 2, 1)
plt.plot(Time, C_A, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$C_{A}$ (mol/L)')
plt.xlim(xlim)

plt.subplot(7, 2, 2)
plt.plot(Time, T, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$T$ (K)')
plt.xlim(xlim)

plt.subplot(7, 2, 3)
plt.plot(Time, T_C, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$T_{C}$ (K)')
plt.xlim(xlim)

plt.subplot(7, 2, 4)
plt.plot(Time, h, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$H$ (m)')
plt.xlim(xlim)

plt.subplot(7, 2, 5)
plt.plot(Time, Q_SP, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$Q_{SP}$ (L/s)')
plt.xlim(xlim)

plt.subplot(7, 2, 6)
plt.plot(Time, Qc_SP, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$Q_{C,SP}$ (L/s)')
plt.xlim(xlim)

plt.subplot(7, 2, 7)
plt.plot(Time, l, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$l$ (% Open)')
plt.xlim(xlim)

plt.subplot(7, 2, 8)
plt.plot(Time, l_c, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$l_{C}$ (% Open)')
plt.xlim(xlim)

plt.subplot(7, 2, 9)
plt.plot(Time, Q, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$Q$ (L/s)')
plt.xlim(xlim)

plt.subplot(7, 2, 10)
plt.plot(Time, Q_c, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$Q_{C}$ (L/s)')
plt.xlim(xlim)

plt.subplot(7, 2, 11)
plt.plot(Time, T_F, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$T_{F}$ (K)')
plt.xlim(xlim)

plt.subplot(7, 2, 12)
plt.plot(Time, C_AF, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$C_{A,F}$ (mol/L)')
plt.xlim(xlim)

plt.subplot(7, 2, 13)
plt.plot(Time, T_CF, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$T_{C,F}$ (K)')
plt.xlim(xlim)

plt.subplot(7, 2, 14)
plt.plot(Time, Q_F, color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$Q_{F}$ (L/s)')
plt.xlim(xlim)



# %%
# zoomed in plots
idx_start = 530*3600
idx_end = 600*3600

plot_var_red = plot_var[idx_start:idx_end]

Time_red = Time[idx_start:idx_end]
C_A_red = C_A[idx_start:idx_end]
T_red = T[idx_start:idx_end]
T_C_red = T_C[idx_start:idx_end]
h_red = h[idx_start:idx_end]
Q_SP_red = Q_SP[idx_start:idx_end]
Qc_SP_red = Qc_SP[idx_start:idx_end]
l_red = l[idx_start:idx_end]
l_c_red = l_c[idx_start:idx_end]
Q_red = Q[idx_start:idx_end]
Q_c_red = Q_c[idx_start:idx_end]
T_F_red = T_F[idx_start:idx_end]

plt.figure(figsize=(6, 4))
plt.plot(Time_red, plot_var_red,color = 'black')
plt.xlabel('Time (h)')
plt.ylabel(plot_var_name)


plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(Time_red, C_A_red,color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$C_{A}$ (mol/L)')
plt.ylim([np.mean(C_A_red)*0.7, np.mean(C_A_red)*1.3])

plt.subplot(3, 2, 2)
plt.plot(Time_red, h_red,color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('H (m)')
plt.ylim([np.mean(h_red)*0.95, np.mean(h_red)*1.05])

plt.subplot(3, 2, 3)
plt.plot(Time_red, T_red,color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('T (K)')
plt.ylim([np.mean(T_red)*0.95, np.mean(T_red)*1.05])

plt.subplot(3, 2, 4)
plt.plot(Time_red, T_C_red,color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$T_{C}$ (K)')
plt.ylim([np.mean(T_C_red)*0.9, np.mean(T_C_red)*1.1])

plt.subplot(3, 2, 5)
plt.plot(Time_red, Q_red,color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('Q (L/s)')
plt.ylim([np.mean(Q_red)*0.8, np.mean(Q_red)*1.1])

plt.subplot(3, 2, 6)
plt.plot(Time_red, Q_c_red,color = 'black')
plt.xlabel('Time (h)')
plt.ylabel('$Q_{c}$ (L/s)')
plt.ylim([np.mean(Q_c_red)*0.6, np.mean(Q_c_red)*1.5])



