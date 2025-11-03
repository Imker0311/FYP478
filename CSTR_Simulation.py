# %%
from scipy.integrate import solve_ivp
import numpy as np
import h5py
import os

# Sun simulation in 'num_steps' number of chunks, 'run' starts from one up to 'num_steps' each run. Set 'run' and 'num_steps' accordingly for each run
# Set 'num_steps' to and 'run' 1 to run full simulation in one go
# the length of the input vectors divided by 'num_steps' should be a multiple of 100 000.
run = 1
num_steps = 3

# File to load input vectors from
file_vectors = 'InputVectors_Test.h5'
# File to save simulation results to
file_save = 'CSTR_Simulation_Testt.h5'

def load_vector(filename,name):
    with h5py.File(filename, 'r') as f:
        data = f[name][:]
        print(f"Loaded dataset '{name}' with shapes {data.shape}")
        return data
    
def save_vector(filename, name, data): # Creates dataset if it does not exist, overwrites if it does
    with h5py.File(filename, 'a') as f:  # open file in append mode
        if name in f:
            del f[name]  # delete old dataset before overwriting
        f.create_dataset(name, data=data, chunks=True, compression='gzip')
        print(f"Saved dataset '{name}' with shape {data.shape}")

def save_or_append_vector(filename, dataset_name, new_data):

    new_data = np.asarray(new_data)
    with h5py.File(filename, 'a') as f:
        if dataset_name in f:
            dset = f[dataset_name]
            old_len = dset.shape[0]
            new_len = old_len + len(new_data)
            dset.resize(new_len, axis=0)
            dset[old_len:new_len] = new_data
        else:
            f.create_dataset(dataset_name, data=new_data, maxshape=(None,1), chunks=True)

def load_partial_vector(filename, name, start=None, end=None):
    with h5py.File(filename, 'r') as f:
        if name not in f:
            raise KeyError(f"Dataset '{name}' not found.")
        dataset = f[name]
        # If start/end not specified, load full dataset
        if start is None and end is None:
            return dataset[:]
        return dataset[start:end]

if run == 1 and os.path.exists(file_save):
    os.remove(file_save)

# initial conditions
T0 = 401.7     # K
T_C0 = 345.2   # K
C_A0 = 0.0453    # mol/L
h0 = 0.6        # m

Q_N = 100/60    # L/s
Q_c_N = 15/60   # L/s

l_B = 0.5       # -
l_c_B = 0.5    # -
Q_B = 1.66     # L/s
Q_c_B = 0.25   # L/s

# parameters
A = 0.1666      # m^2
k_0 = 7.2e10/60 # 1/s
dH = -5e4       # J/mol
rho_Cp =  239   # J/L K
rhoc_Cpc = 4175 # J/L K
V_c = 10        # L

Cv1 = 0.4714    #L/s Psi^1/2
Cv2 = 0.1       #L/s Psi^1/2

E_R = 8750      # K
U_Ac = 5e4/60   # J/s K
T_bias = 0      # K
T_F = 320       # K
C_AF = 1      # mol/L
T_CF = 300      # K
Q_F = Q_N       # L/s
dP = 50         # Psi
dPc = 25        # Psi

class controller:
    def __init__(self, Kc , Ti, B, bounds):

        self.Kc = Kc
        self.Ti = Ti
        self.B = B
        self.bounds = bounds
        self.P = np.array([])
        self.I = np.array([])
        self.E = np.array([])
        self.Sum_E = np.array([0])
        self.Sum_E_prev = np.array([0])

    def PI(self, t, MV, CV, SP):

        dt = 1
                
        if t >= 1:
                    
            # error
            self.E = (SP - CV)
                    
            # sum of error

            self.Sum_E = self.Sum_E_prev + self.E*dt

            #anti-windup logic
            if MV == self.bounds[0] or MV == self.bounds[1]:
                self.Sum_E = self.Sum_E_prev

            self.Sum_E_prev = self.Sum_E

            # PI controller
            self.P = self.E
            self.I = ((1/self.Ti) * self.Sum_E)

            PI = self.Kc * (self.P + self.I) + self.B
            r = max(self.bounds[0], min(self.bounds[1], PI))      #clip

        else:

            r = self.B

        return r

class CSTR:
    def __init__(self):
        return
    
    def ODE(self, t, y):

        C_A = y[0]
        T = y[1]
        T_C = y[2]
        h = y[3]
        l_actual = y[4]
        l_C_actual = y[5]

        # System
        dCAdt = (Q_F*C_AF - Q*C_A)/(1000*A*h) - k_0 * np.exp(-E_R/T) * C_A
        dTdt = (k_0 * np.exp(-E_R/T) * C_A * (-dH))/rho_Cp + (Q_F*T_F - Q*T)/(1000*A*h) + (U_Ac*(T_C - T))/(rho_Cp*1000*A*h)
        dTcdt = Q_c*(T_CF - T_C)/V_c + U_Ac*(T - T_C)/(rhoc_Cpc*V_c)
        dhdt = (Q_F - Q)/(1000*A)

        # Valve dynamics
        dldt = (l[-1] - l_actual)/(2)
        dlcdt = (l_c[-1] - l_C_actual)/(2)

        return np.array([dCAdt, dTdt, dTcdt, dhdt, dldt, dlcdt])

###### Controllers ######
# Outlet Flowrate setpoint
Q_SP_P = controller(-30, 80, Q_B, np.array([0, Cv1*np.sqrt(dP)]))
Q_SP = np.array([Q_B])

# Outlet Flowrate Valve
l_P = controller(0.2, 1, l_B, np.array([0, 1]))
l = np.array([l_B])

# Cooling Flowrate setpoint
Qc_SP_P = controller(-0.01, 250, Q_c_B, np.array([0, Cv2*np.sqrt(dPc)]))
Qc_SP = np.array([Q_c_B])

# Cooling Flowrate Valve
l_C_P = controller(0.2, 1, l_c_B, np.array([0, 1]))
l_c = np.array([l_c_B])

system = CSTR()
# Initialise solver variables
y0 = [C_A0, T0, T_C0, h0, l_B, l_c_B]

# Solution arrays
Time = np.array([0])
Y = np.array([y0])

# Initialise Flowrates
Q = Q_N
Q_c = Q_c_N
Q_vec = np.array([Q])
Q_c_vec = np.array([Q_c])

if run > 1:
    Initial_Conditions = load_vector('Initial_Conditions.h5', 'Initial_Conditions')

    y0 = Initial_Conditions[6:12]
    Q_SP = np.array([Initial_Conditions[0]])
    l = np.array([Initial_Conditions[1]])
    Qc_SP = np.array([Initial_Conditions[2]])
    l_c = np.array([Initial_Conditions[3]])
    Q_vec = np.array([Initial_Conditions[4]])
    Q_c_vec = np.array([Initial_Conditions[5]])
    t_prev = Initial_Conditions[12]

# Time
t = load_vector(file_vectors,'t')
t_eval = t[(run-1)*(len(t)-1)//num_steps+1:((run)*(len(t)-1)//num_steps)+1]
t_prev = (run-1)*(len(t)-1)//num_steps
print(f't_prev:{t_prev}')

E_R_vec = load_partial_vector(file_vectors, 'F1', (run-1)*(len(t)-1)//num_steps+1, (run)*(len(t)-1)//num_steps+1)
U_Ac_vec = load_partial_vector(file_vectors, 'F2', (run-1)*(len(t)-1)//num_steps+1, (run)*(len(t)-1)//num_steps+1)
T_bias_vec = load_partial_vector(file_vectors, 'F3', (run-1)*(len(t)-1)//num_steps+1, (run)*(len(t)-1)//num_steps+1)
T_F_vec = load_partial_vector(file_vectors, 'F4', (run-1)*(len(t)-1)//num_steps+1, (run)*(len(t)-1)//num_steps+1)
C_AF_vec = load_partial_vector(file_vectors, 'F5', (run-1)*(len(t)-1)//num_steps+1, (run)*(len(t)-1)//num_steps+1)
T_CF_vec = load_partial_vector(file_vectors, 'F6', (run-1)*(len(t)-1)//num_steps+1, (run)*(len(t)-1)//num_steps+1)
Q_F_vec = load_partial_vector(file_vectors, 'F7', (run-1)*(len(t)-1)//num_steps+1, (run)*(len(t)-1)//num_steps+1)
dP_vec = load_partial_vector(file_vectors, 'F8', (run-1)*(len(t)-1)//num_steps+1, (run)*(len(t)-1)//num_steps+1)
dPc_vec = load_partial_vector(file_vectors, 'F9', (run-1)*(len(t)-1)//num_steps+1, (run)*(len(t)-1)//num_steps+1)

i = 0

for t in t_eval:

    print(t)

    E_R = E_R_vec[i,0]
    U_Ac = U_Ac_vec[i,0]
    T_bias = T_bias_vec[i,0]
    T_F = T_F_vec[i,0]
    C_AF = C_AF_vec[i,0]
    T_CF = T_CF_vec[i,0]
    Q_F = Q_F_vec[i,0]
    dP = dP_vec[i,0]
    dPc = dPc_vec[i,0]

    # ODE Solver
    sol = solve_ivp(system.ODE, (t_prev, t), y0)
    t_prev = t
    y0 = sol.y[:,-1]

    i += 1

    # Controllers
    Q_SP = np.append(Q_SP, Q_SP_P.PI(t, Q_SP[-1], y0[3], h0))
    l = np.append(l, l_P.PI(t, l[-1], Q_vec[-1], Q_SP[-1]))
    Qc_SP = np.append(Qc_SP, Qc_SP_P.PI(t, Qc_SP[-1], y0[1]+T_bias, T0))
    l_c = np.append(l_c, l_C_P.PI(t, l_c[-1], Q_c_vec[-1], Qc_SP[-1]))

    # Flowrates
    Q = Cv1*np.sqrt(dP)*y0[4]
    Q_vec = np.append(Q_vec, Q)
    Q_c = Cv2*np.sqrt(dPc)*y0[5]
    Q_c_vec = np.append(Q_c_vec, Q_c)

    # Store solution
    Y = np.vstack((Y, y0.reshape(1,-1)))
    Time = np.append(Time, sol.t[-1])

    if t % 50000 < 1e-8:

        Initial_Conditions = np.zeros(15)

        #controllers
        save_or_append_vector(file_save, 'Q_SP', Q_SP[1:].reshape(-1,1))
        a = Q_SP[-1]
        del Q_SP
        Q_SP = np.array([a])
        Initial_Conditions[0] = a

        save_or_append_vector(file_save, 'l', l[1:].reshape(-1,1))
        a = l[-1]
        del l
        l = np.array([a])
        Initial_Conditions[1] = a

        save_or_append_vector(file_save, 'Qc_SP', Qc_SP[1:].reshape(-1,1))
        a = Qc_SP[-1]
        del Qc_SP
        Qc_SP = np.array([a])
        Initial_Conditions[2] = a

        save_or_append_vector(file_save, 'l_c', l_c[1:].reshape(-1,1))
        a = l_c[-1]
        del l_c
        l_c = np.array([a])
        Initial_Conditions[3] = a

        #flowrates
        save_or_append_vector(file_save, 'Q_vec', Q_vec[1:].reshape(-1,1))
        a = Q_vec[-1]
        del Q_vec
        Q_vec = np.array([a])
        Initial_Conditions[4] = a

        save_or_append_vector(file_save, 'Q_c_vec', Q_c_vec[1:].reshape(-1,1))
        a = Q_c_vec[-1]
        del Q_c_vec
        Q_c_vec = np.array([a])
        Initial_Conditions[5] = a

        #solution and time
        save_or_append_vector(file_save, 'C_A', Y[1:,0].reshape(-1,1))
        save_or_append_vector(file_save, 'T', Y[1:,1].reshape(-1,1))
        save_or_append_vector(file_save, 'T_C', Y[1:,2].reshape(-1,1))
        save_or_append_vector(file_save, 'h', Y[1:,3].reshape(-1,1))

        a = Y[-1,0]
        b = Y[-1,1]
        c = Y[-1,2]
        d = Y[-1,3]
        e = Y[-1,4]
        f = Y[-1,5]
        del Y
        Y = np.array([[a, b, c, d, e, f]])

        Initial_Conditions[6] = a
        Initial_Conditions[7] = b
        Initial_Conditions[8] = c
        Initial_Conditions[9] = d
        Initial_Conditions[10] = e
        Initial_Conditions[11] = f

        save_or_append_vector(file_save, 'Time', Time[1:].reshape(-1,1))
        a = Time[-1]
        del Time
        Time = np.array([a])
        Initial_Conditions[12] = a
        print("Saved data at time:", t)

        save_vector('Initial_Conditions.h5', 'Initial_Conditions', Initial_Conditions)
        





