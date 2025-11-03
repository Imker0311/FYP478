# %%
import numpy as np
import h5py
from scipy.interpolate import interp1d

file = 'InputVectors_Test.h5'   # Save input vectors to this file.

class fault:
    def __init__(self, t, x, y, faultlength, Variance, type, P1, P2, sign): #Type: 0-bathtub,1-weibull,2-uniform,3-exponential, P1 and P2 are parameters for the distributions, sign: 0-positive only, 1-± directions
        
        self.faultlength = faultlength
        self.Var = Variance
        self.x = x
        self.y = y
        self.t = np.array(t)
        
        self.type = type
        self.P1 = P1
        self.P2 = P2
        self.sign = sign

        self.idx = np.array([0])
        self.idxend = np.array([0])
        self.plt = np.ones((len(self.t), 1))
        self.active = False
        self.multiplier = np.array([])

    def probability(self, t, RNG):

        tidx = np.where(self.t == t)[0][0]

        t_now = t - self.t[self.idxend[-1]]     # Reset time used in P after fault
        t_prev = self.t[tidx-1] - self.t[self.idxend[-1]]

        match self.type:
            case 0: #bath-tub P1:b, P2:L

                P = 16*(1 - self.P1*self.P2)/self.P2**5 * ((t_now - self.P2/2)**5 + (self.P2/2)**5) + self.P1*(t_now)
                P_prev = 16*(1 - self.P1*self.P2)/self.P2**5 * ((t_prev - self.P2/2)**5 + (self.P2/2)**5) + self.P1*t_prev
                h = (P - P_prev)/(1-P_prev)

            case 1: #weibull P1:B, P2:n

                P = 1 - np.exp(-((t_now/self.P2)**self.P1))
                P_prev = 1 - np.exp(-((t_prev/self.P2)**self.P1))
                h = (P - P_prev)/(1-P_prev)

            case 2: # uniform P1:-, P2: L

                P = 1/self.P2 * t_now
                P_prev = 1/self.P2 * t_prev
                h = (P - P_prev)/(1-P_prev)

            case 3: # exponential (not used yet)

                P = 0.4
                h = 0.4

        if h > RNG:
            # fault is active
            self.active = True

            self.idx = np.append(self.idx, np.where(self.t == t)[0][0])

            self.idxend = np.append(self.idxend, self.idx[-1] + self.faultlength)

            # random magnitude between 25 and 125 % with random sign if needed
            self.multiplier = (1 + (np.random.random() * self.Var)) * np.random.choice([1, -1]) if self.sign == 1 else (1 + (np.random.random() * self.Var))
        else:
            # fault is inactive
            self.active = False

        return 

    def vector_update(self):

        for time in self.t:
            tidx = np.where(self.t == time)[0][0]

            if tidx > self.idxend[-1]:
                RNG = np.random.random()
                fault.probability(self, time, RNG)
                self.plt[tidx] = 0 #np.append(self.plt, 0)
                print(time, k)

                if self.active == True:

                    # update the fault vector
                    if self.idxend[-1] > np.where(self.t == self.t[-1])[0][0]:
                        self.idx = self.idx[:-1]
                        self.active = False
                        
                    else:
                        self.x[self.idx[-1]:self.idxend[-1]] = (self.y * self.multiplier) + self.x[1]

                        self.active = False

            else:

                self.plt[tidx] = 1 if time > 0 else 0 #np.append(self.plt, 1) if time>0 else 0
            
        return self.x
    
def upsample(t, y, factor):

    f = interp1d(t.reshape(-1,), y.reshape(-1,), kind='linear')
    t_new = np.linspace(t[0], t[-1], (len(t)-1)*factor+1)
    y_new = f(t_new)

    return y_new.reshape(-1, 1)

def AR1(x, p, s):
    
    AR = np.zeros(((len(x)),1))
    i = 1

    for t in np.linspace(0, 1, len(x)-1):

        AR[i] = AR[i-1]*p + np.random.normal(0, s)
        i += 1
    
    return AR/100 + 1

def save_vector(filename, name, data): # Creates dataset if it does not exist, overwrites if it does
    with h5py.File(filename, 'a') as f:  # open file in append mode
        if name in f:
            del f[name]  # delete old dataset before overwriting
        f.create_dataset(name, data=data, chunks=True, compression='gzip')
        print(f"Saved dataset '{name}' with shape {data.shape}")

t = np.linspace(0, 25000, 25001) # timesteps
factor = 60                      # number of seconds per timestep
sc = 0.7                         # Scale Probability Time Constant, larger value = less frequent faults

t_step = t[1] - t[0]

# AR1 parameters
p = 0.9999
s = 0.005

range = (1,2,3,4,5,6,7,8,9)

for run in range:

    match run:
        case 1:
            
            k = 1
            F1_length = 300 # length of fault in steps 
            E_R = 8750 * np.ones((len(t), 1)) #F_1
            F_1_vec = ((0.3) * np.linspace(0,F1_length*t_step, F1_length)).reshape(-1, 1) #E/R (catalyst deactivation)
            Variance = 0.1 # variance of fault magnitude

            F_1 = fault(t, E_R, F_1_vec, F1_length, Variance, 1, 4, sc*20000, 0)      #weibull distribution
            F_1.vector_update()
            
            F1_upsample = upsample(t, F_1.x, factor)
            F1_vec = F1_upsample

            save_vector(file,'F1', F1_vec)
            save_vector(file,'F1.plt', np.insert(np.repeat(F_1.plt[1:],factor),0,0))

            del F_1
            del F1_vec
            del F1_upsample
            del E_R

        case 2:

            k = 2
            F2_length = 360 # length of fault in steps
            U_Ac = 5e4/60 * np.ones((len(t), 1)) #F_2
            F_2_vec = ((-15/60) * np.linspace(0,F2_length*t_step, F2_length)).reshape(-1, 1) #U_Ac (HX fouling)
            Variance = 0.1 # variance of fault magnitude

            F_2 = fault(t, U_Ac, F_2_vec, F2_length, Variance, 1, 4, sc*20000 , 0)      #weibull distribution 
            F_2.vector_update()

            F2_upsample = upsample(t, F_2.x, factor)
            F2_vec = F2_upsample

            save_vector(file,'F2', F2_vec)
            save_vector(file,'F2.plt', np.insert(np.repeat(F_2.plt[1:],factor),0,0))

            del F_2
            del F2_vec
            del F2_upsample
            del U_Ac

        case 3:

            k = 3
            F3_length = 30 # length of fault in steps
            T_Bias = np.zeros((len(t), 1)) #F_3
            F_3_vec = 5 * np.ones((F3_length, 1)) #T Bias (Sensor Bias)
            Variance = 0.5 # variance of fault magnitude

            F_3 = fault(t, T_Bias, F_3_vec, F3_length, Variance, 1, 2, sc*12000 , 1)      #weibull distribution
            F_3.vector_update()

            F3_upsample = upsample(t, F_3.x, factor)
            F3_vec = F3_upsample * AR1(F3_upsample, p, s)

            save_vector(file,'F3', F3_vec)
            save_vector(file,'F3.plt', np.insert(np.repeat(F_3.plt[1:],factor),0,0))

            del F_3
            del F3_vec
            del F3_upsample
            del T_Bias 

        case 4:

            k = 4
            F4_length = 90 # length of fault in steps
            T_F = 320 * np.ones((len(t), 1)) #F_4
            F_4_vec = ((0.35) * np.linspace(0,F4_length*t_step, F4_length)).reshape(-1, 1) #T_F (change in feed temperature)
            Variance = 0.2 # variance of fault magnitude

            F_4 = fault(t, T_F, F_4_vec, F4_length, Variance, 1, 3, sc*17000 , 1)      #weibull distribution
            F_4.vector_update()

            F4_upsample = upsample(t, F_4.x, factor)
            F4_vec = F4_upsample * AR1(F4_upsample, p, s)

            save_vector(file,'F4', F4_vec)
            save_vector(file,'F4.plt', np.insert(np.repeat(F_4.plt[1:],factor),0,0))

            del F_4
            del F4_vec
            del F4_upsample
            del T_F

        case 5:

            k = 5
            F5_length = 90 # length of fault in steps
            C_F = 1 * np.ones((len(t), 1)) #F_5
            F_5_vec = ((0.0006) * np.linspace(0,F5_length*t_step, F5_length)).reshape(-1, 1) #C_F (change in feed concentration)
            Variance = 0.3 # variance of fault magnitude

            F_5 = fault(t, C_F, F_5_vec, F5_length, Variance, 1, 2, sc*19000 , 1)      #weibull distribution
            F_5.vector_update()

            F5_upsample = upsample(t, F_5.x, factor)
            F5_vec = F5_upsample * AR1(F5_upsample, p, s)

            save_vector(file,'F5', F5_vec)
            save_vector(file,'F5.plt', np.insert(np.repeat(F_5.plt[1:],factor),0,0))

            del F_5
            del F5_vec
            del F5_upsample
            del C_F

        case 6:

            k = 6
            F6_length = 150 # length of fault in steps
            T_CF = 300 * np.ones((len(t), 1)) #F_6
            F_6_vec = ((0.1) * np.linspace(0,F6_length*t_step, F6_length)).reshape(-1, 1) #T_CF
            Variance = 0.2 # variance of fault magnitude

            F_6 = fault(t, T_CF, F_6_vec, F6_length, Variance, 1, 4, sc*17000 , 1)      #weibull distribution
            F_6.vector_update()

            F6_upsample = upsample(t, F_6.x, factor)
            F6_vec = F6_upsample * AR1(F6_upsample, p, s)
            save_vector(file,'F6', F6_vec)
            save_vector(file,'F6.plt', np.insert(np.repeat(F_6.plt[1:],factor),0,0))

            del F_6
            del F6_vec
            del F6_upsample
            del T_CF

        case 7:

            k = 7
            F7_length = 60 # length of fault in steps
            Q_F = (100/60) * np.ones((len(t), 1)) # F_7
            F_7_vec = (10/60) * np.ones((F7_length, 1)) #Q_F (change in feed flow rate)
            Variance = 0.1 # variance of fault magnitude

            F_7 = fault(t, Q_F, F_7_vec, F7_length, Variance, 1, 2, sc*15000, 1)    #uniform distribution
            F_7.vector_update()

            F7_upsample = upsample(t, F_7.x, factor)
            F7_vec = F7_upsample * AR1(F7_upsample, p, s)

            save_vector(file,'F7', F7_vec)
            save_vector(file,'F7.plt', np.insert(np.repeat(F_7.plt[1:],factor),0,0))

            del F_7
            del F7_vec
            del F7_upsample
            del Q_F

        case 8:

            k = 8
            F8_length = 60 # length of fault in steps
            dP = 50 * np.ones((len(t), 1)) #F_8
            F_8_vec = (5) * np.ones((F8_length, 1)) #dP (change in pressure drop of reactor outlet)
            Variance = 0.4 # variance of fault magnitude

            F_8 = fault(t, dP, F_8_vec, F8_length, Variance, 1, 2, sc*19000 , 1)      #weibull distribution
            F_8.vector_update()

            F8_upsample = upsample(t, F_8.x, factor)
            F8_vec = F8_upsample * AR1(F8_upsample, p, s)

            save_vector(file,'F8', F8_vec)
            save_vector(file,'F8.plt', np.insert(np.repeat(F_8.plt[1:],factor),0,0))

            del F_8
            del F8_vec
            del F8_upsample
            del dP

        case 9:
            k = 9
            F9_length = 60 # length of fault in steps
            dPc = 25 * np.ones((len(t), 1)) #F_9
            F_9_vec = (2.5) * np.ones((F9_length, 1)) #dPc (change in pressure drop of coolent)
            Variance = 0.4 # variance of fault magnitude

            F_9 = fault(t, dPc, F_9_vec, F9_length, Variance, 1, 2, sc*19000 , 1)      #weibull distribution
            F_9.vector_update()

            F9_upsample = upsample(t, F_9.x, factor)
            F9_vec = F9_upsample * AR1(F9_upsample, p, s)

            save_vector(file,'F9', F9_vec)
            save_vector(file,'F9.plt', np.insert(np.repeat(F_9.plt[1:],factor),0,0))

            del F_9
            del F9_vec
            del F9_upsample
            del dPc

save_vector(file,'t', np.linspace(t[0]*factor, (t[-1])*factor, (len(t)-1)*factor+1))




