# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd

def load_vector(filename, name):
    with h5py.File(filename, 'r') as f:
        data = f[name][:]
        print(f"Loaded dataset '{name}' with shape {data.shape}")
        return data

def save_vector(filename, name, data): # Creates dataset if it does not exist, overwrites if it does
    with h5py.File(filename, 'a') as f:  # open file in append mode
        if name in f:
            del f[name]  # delete old dataset before overwriting
        f.create_dataset(name, data=data, chunks=True, compression='gzip')
        print(f"Saved dataset '{name}' with shape {data.shape}")

#File containing simulation results of training data
file = 'CSTR_Simulation_Train.h5'

C_A = load_vector(file, 'C_A')
T = load_vector(file, 'T')
T_C = load_vector(file, 'T_C')
h = load_vector(file, 'h')
Q = load_vector(file, 'Q_vec')
Q_C = load_vector(file, 'Q_c_vec')
l = load_vector(file, 'l')
Q_SP = load_vector(file, 'Q_SP')
l_C = load_vector(file, 'l_c')
Q_C_SP = load_vector(file, 'Qc_SP')

# File containing input vectors of training data
file = 'InputVectors_Train.h5'

t = load_vector(file, 't')
T_F = load_vector(file, 'F4')
C_AF = load_vector(file, 'F5')
T_CF = load_vector(file, 'F6')
Q_F = load_vector(file, 'F7')

F1 = load_vector(file, 'F1.plt').reshape(-1)*1
F2 = load_vector(file, 'F2.plt').reshape(-1)*2
F3 = load_vector(file, 'F3.plt').reshape(-1)*3
F4 = load_vector(file, 'F4.plt').reshape(-1)*4
F5 = load_vector(file, 'F5.plt').reshape(-1)*5
F6 = load_vector(file, 'F6.plt').reshape(-1)*6
F7 = load_vector(file, 'F7.plt').reshape(-1)*7
F8 = load_vector(file, 'F8.plt').reshape(-1)*8
F9 = load_vector(file, 'F9.plt').reshape(-1)*9

T_F = T_F[1:]
C_AF = C_AF[1:]
T_CF = T_CF[1:]
Q_F = Q_F[1:]
t = t[1:]
F1 = F1[1:]
F2 = F2[1:]
F3 = F3[1:]
F4 = F4[1:]
F5 = F5[1:]
F6 = F6[1:]
F7 = F7[1:]
F8 = F8[1:]
F9 = F9[1:]


def combine_fault_vectors(faults):

    active_counts = np.count_nonzero(faults, axis=0)
    
    combined = np.zeros(faults.shape[1], dtype=int)
    
    single_fault_idx = np.where(active_counts == 1)[0]

    combined[single_fault_idx] = faults[:, single_fault_idx].sum(axis=0)

    multiple_fault_idx = np.where(active_counts > 1)[0]

    combined[multiple_fault_idx] = 10
    
    return combined

faults = np.vstack([F1,F2,F3,F4,F5,F6,F7,F8,F9])
Fault_class = combine_fault_vectors(faults).reshape(-1, 1)
Fault_class_reduced = Fault_class[::20, :]

df_full = np.hstack([Fault_class, C_A, T, T_C, h, Q, Q_C, Q_F, C_AF, T_F, T_CF, l, Q_SP, l_C, Q_C_SP])
df = df_full[::20, :]

values, counts = np.arange(11), np.bincount(Fault_class_reduced.ravel(), minlength=11)
for val, count in zip(values, counts):
    print(f"Fault {val}: {count} occurrences")
print(f"Data shape: {df.shape}")

# Select scaler
scaler = 1  # 1 for StandardScaler, 2 for MinMaxScaler, 3 for RobustScaler
match scaler:
    case 1:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(df[df[:, 0] == 0, 1:])
        df_scaled =sc.transform(df[:, 1:])
    case 2:
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))
        sc.fit(df[df[:, 0] == 0, 1:])  # Fit only on normal operation data
        df_scaled = sc.transform(df[:, 1:])
    case 3:
        from sklearn.preprocessing import RobustScaler
        sc = RobustScaler()
        sc.fit(df[df[:, 0] == 0, 1:])  # Fit only on normal operation data
        df_scaled = sc.transform(df[:, 1:])


X = df_scaled
Y = df[:, 0]

# File to save preprocessed data
file = 'Dataframe.h5'

save_vector(file, 'X', X)
save_vector(file, 'Y', Y)

# %%
# File containing simulation results of testing data
file = 'CSTR_Simulation_Test.h5'

C_A = load_vector(file, 'C_A')
T = load_vector(file, 'T')
T_C = load_vector(file, 'T_C')
h = load_vector(file, 'h')
Q = load_vector(file, 'Q_vec')
Q_C = load_vector(file, 'Q_c_vec')
l = load_vector(file, 'l')
Q_SP = load_vector(file, 'Q_SP')
l_C = load_vector(file, 'l_c')
Q_C_SP = load_vector(file, 'Qc_SP')

# File containing input vectors of testing data
file = 'InputVectors_Test.h5'

t = load_vector(file, 't')
T_F = load_vector(file, 'F4')
C_AF = load_vector(file, 'F5')
T_CF = load_vector(file, 'F6')
Q_F = load_vector(file, 'F7')

F1 = load_vector(file, 'F1.plt').reshape(-1)*1
F2 = load_vector(file, 'F2.plt').reshape(-1)*2
F3 = load_vector(file, 'F3.plt').reshape(-1)*3
F4 = load_vector(file, 'F4.plt').reshape(-1)*4
F5 = load_vector(file, 'F5.plt').reshape(-1)*5
F6 = load_vector(file, 'F6.plt').reshape(-1)*6
F7 = load_vector(file, 'F7.plt').reshape(-1)*7
F8 = load_vector(file, 'F8.plt').reshape(-1)*8
F9 = load_vector(file, 'F9.plt').reshape(-1)*9

T_F = T_F[1:]
C_AF = C_AF[1:]
T_CF = T_CF[1:]
Q_F = Q_F[1:]
t = t[1:]
F1 = F1[1:]
F2 = F2[1:]
F3 = F3[1:]
F4 = F4[1:]
F5 = F5[1:]
F6 = F6[1:]
F7 = F7[1:]
F8 = F8[1:]
F9 = F9[1:]

def combine_fault_vectors(faults):

    active_counts = np.count_nonzero(faults, axis=0)
    
    combined = np.zeros(faults.shape[1], dtype=int)
    
    single_fault_idx = np.where(active_counts == 1)[0]

    combined[single_fault_idx] = faults[:, single_fault_idx].sum(axis=0)

    multiple_fault_idx = np.where(active_counts > 1)[0]

    combined[multiple_fault_idx] = 10
    
    return combined

faults = np.vstack([F1,F2,F3,F4,F5,F6,F7,F8,F9])
Fault_class = combine_fault_vectors(faults).reshape(-1, 1)
Fault_class_reduced = Fault_class[::20, :]

df_full = np.hstack([Fault_class, C_A, T, T_C, h, Q, Q_C, Q_F, C_AF, T_F, T_CF, l, Q_SP, l_C, Q_C_SP])
df = df_full[::20, :]

values, counts = np.arange(11), np.bincount(Fault_class_reduced.ravel(), minlength=11)
for val, count in zip(values, counts):
    print(f"Fault {val}: {count} occurrences")
print(f"Data shape: {df.shape}")

df_scaled = sc.transform(df[:, 1:])

X_test = df_scaled
y_test = df[:, 0]

# File to save preprocessed data
file = 'Dataframe.h5'

save_vector(file, 'X_test', X_test)
save_vector(file, 'y_test', y_test)
save_vector(file, 'counts', counts)

df_pds = pd.DataFrame(df_full, columns=['Fault_Class', 'C_A', 'T', 'T_C', 'h', 'Q', 'Q_C', 'Q_F', 'C_AF', 'T_F', 'T_CF', 'l', 'Q_SP', 'l_C', 'Q_C_SP'])


# %%
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

sns.kdeplot(data=df_pds, x='T', hue='Fault_Class', fill=True, common_norm=False, alpha=0.5, ax=axes[0, 0])
axes[0, 0].set_xlabel('Temperature (T)')
axes[0, 0].set_ylabel('Density')

sns.kdeplot(data=df_pds, x='C_A', hue='Fault_Class', fill=True, common_norm=False, alpha=0.5, ax=axes[0, 1])
axes[0, 1].set_xlabel('Concentration (C_A)')
axes[0, 1].set_ylabel('Density')

sns.kdeplot(data=df_pds, x='T_C', hue='Fault_Class', fill=True, common_norm=False, alpha=0.5, ax=axes[1, 0])
axes[1, 0].set_xlabel('Coolant Temperature (T_C)')
axes[1, 0].set_ylabel('Density')

sns.kdeplot(data=df_pds, x='h', hue='Fault_Class', fill=True, common_norm=False, alpha=0.5, ax=axes[1, 1])
axes[1, 1].set_xlabel('Liquid Level (h)')
axes[1, 1].set_ylabel('Density')

sns.kdeplot(data=df_pds, x='Q', hue='Fault_Class', fill=True, common_norm=False, alpha=0.5, ax=axes[2, 0])
axes[2, 0].set_xlabel('Flow Rate (Q)')
axes[2, 0].set_ylabel('Density')

sns.kdeplot(data=df_pds, x='Q_C', hue='Fault_Class', fill=True, common_norm=False, alpha=0.5, ax=axes[2, 1])
axes[2, 1].set_xlabel('Coolant Flow Rate (Q_C)')
axes[2, 1].set_ylabel('Density')

plt.tight_layout()
plt.show()