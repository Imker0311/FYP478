import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

Display = 0

def load_vector(filename,name):
    with h5py.File(filename, 'r') as f:
        data = f[name][:]
        #print(f"Loaded dataset '{name}' with shapes {data.shape}")
        return data

def save_vector(filename, name, data): # Creates dataset if it does not exist, overwrites if it does
    with h5py.File(filename, 'a') as f:  # open file in append mode
        if name in f:
            del f[name]  # delete old dataset before overwriting
        f.create_dataset(name, data=data, chunks=True, compression='gzip')
        #print(f"Saved dataset '{name}' with shape {data.shape}")

class metrics:
    def __init__(self, y_test, y_pred, counts):
        self.y_test = y_test
        self.y_pred = y_pred
        self.counts = counts[0:10]

        self.undetected = False
        self.Fault_starts = np.array([])
        self.Fault_detected = np.array([])

        self.fault = False
        self.counter = 0
        self.correct = 0
        self.FPcounter = 0
        self.FP0counter = 0
        self.FPidx = 0
        self.FP0idx = 0

        self.n = 10

        mask = (self.y_test != 10) & (self.y_pred != 10)
        self.y_pred = self.y_pred[mask]
        self.y_test = self.y_test[mask]

        self.indices = np.array([])
        self.faultendidx = np.array([])
        self.counteridx = np.array([])

        self.fault_count = np.zeros(9)
        self.fault_durations = np.array([300,360,30,90,90,150,60,60,60])*60

        self.vector = 0

    def table(self):

        self.vector = [
        self.counter,
        self.correct,
        self.FPcounter,
        self.FP0counter,
        int(np.nanmean(self.Ave_time[1:])),
        self.Ave_time[1], self.Ave_time[2], self.Ave_time[3],
        self.Ave_time[4], self.Ave_time[5], self.Ave_time[6],
        self.Ave_time[7], self.Ave_time[8], self.Ave_time[9]]

        if Display == 1:
            # Labels for each column
            labels = [
                "Counter", "Correct", "FPcounter", "FP0counter", "AverageTime",
                "Fault1", "Fault2", "Fault3", "Fault4", "Fault5", "Fault6", "Fault7", "Fault8", "Fault9"]

            df = pd.DataFrame([self.vector], columns=labels)
            display(df)
        
    def metrics(self):

        EM = np.zeros((self.n,self.n))

        for j in range(self.n):
            for i in range(len(self.y_test)):
                if self.y_test[i] == j:
                    if self.y_pred[i] == j:
                        EM[j,j] = EM[j,j] + 1
                    else:
                        EM[j,int(self.y_pred[i])] = EM[j,int(self.y_pred[i])] + 1

        EM = EM.astype(int)
        np.set_printoptions(suppress=True)
        np.set_printoptions(linewidth=120)

        Sensitivity = np.zeros(self.n)
        Specificity = np.zeros(self.n)
        total_sum = np.sum(EM)
        Precision = np.zeros(self.n)

        for j in range(self.n):
            Sensitivity[j] = EM[j,j]/np.sum(EM[j,:])

            TP = EM[j, j]
            FP = np.sum(EM[:, j]) - TP
            FN = np.sum(EM[j, :]) - TP
            TN = total_sum - (TP + FP + FN)
            
            Specificity[j] = TN / (TN + FP) if (TN + FP) > 0 else 0
            Precision[j] = TP / (TP + FP) if (TP + FP) > 0 else 0

        Ave_Sensitivity = np.sum(self.counts[0:11]*Sensitivity[0:11])/np.sum(self.counts[0:11])
        Ave_Specificity = np.sum(self.counts[0:11]*Specificity[0:11])/np.sum(self.counts[0:11])
        Ave_Precision = np.sum(self.counts[0:11]*Precision[0:11])/np.sum(self.counts[0:11])

        Ave_Sensitivity_Ex = np.sum(self.counts[1:11]*Sensitivity[1:11])/np.sum(self.counts[1:11])
        Ave_Specificity_Ex = np.sum(self.counts[1:11]*Specificity[1:11])/np.sum(self.counts[1:11])
        Ave_Precision_Ex = np.sum(self.counts[1:11]*Precision[1:11])/np.sum(self.counts[1:11])
        
        if Display == 1:
            print('Class:  ', 'Sensitivity:', 'Specificity:', 'Precision:')
            for j in range(len(Sensitivity)):
                print(f'{j}       ', f'{Sensitivity[j]:.3f}       ', f'{Specificity[j]:.3f}       ', f'{Precision[j]:.3f}      ')
            print(f'Ave      ', f'{Ave_Sensitivity:.3f}       ', f'{Ave_Specificity:.3f}       ', f'{Ave_Precision:.3f}      ')
            print(f'ExclF0   ', f'{Ave_Sensitivity_Ex:.3f}       ', f'{Ave_Specificity_Ex:.3f}       ', f'{Ave_Precision_Ex:.3f}      ')
            print('\nF1 Score:', f'{(2*(Ave_Precision*Ave_Sensitivity)/(Ave_Precision+Ave_Sensitivity)):.3f}')
            print('\nConfusion Matrix:')
            print(EM)
    
    def detection_time(self):

        for i in range(1,len(self.y_pred)):
            a = self.y_test[i] - self.y_test[i-1]
            diff = self.y_test[i] - self.y_pred[i]

            if a != 0 and self.y_test[i] > 0:
                fault = self.y_test[i]
                self.undetected = True
                idx_start = i
                start_info = np.array([[fault, idx_start]])
                self.Fault_starts = np.vstack((self.Fault_starts, start_info)) if self.Fault_starts.size else start_info
            
            if self.undetected and self.y_test[i] == 0:
                self.undetected = False  # reset if fault clears without detection
                start_info = start_info[:-1]
                self.Fault_starts = self.Fault_starts[:-1,:]

            if diff == 0 and self.undetected and self.y_pred[i+1] == self.y_pred[i]:  # needs two consecutive correct predictions to count
                self.undetected = False
                idx_end = i
                end_info = np.array([fault, idx_end, (idx_end - idx_start)*20])
                self.Fault_detected = np.vstack((self.Fault_detected, end_info)) if self.Fault_detected.size else end_info

        for i in range(1, len(self.y_pred)):
            if self.y_test[i] > 0 and self.y_test[i] != self.y_test[i-1]:
                self.fault_count[int(self.y_test[i])-1] += 1
            
        Ave_times = np.zeros(self.n)
        for j in range(1,self.n):
            times = []
            for i in range(len(self.Fault_detected)):
                if self.Fault_detected[i,0] == j:
                    times.append(self.Fault_detected[i,2])

            if len(times)<self.fault_count[j-1]:
                for m in range(0,int(self.fault_count[j-1])-len(times)):
                    times.append(self.fault_durations[j-1])
            
            if len(times) > 0:
                Ave_times[j] = np.mean(times)
            else:
                Ave_times[j] = np.nan

        if Display == 1:
            print("\n=== Fault Detection Times ===")
            print("📌 Fault Starts (Fault, Start Index):")
            for fault, start in self.Fault_starts:
                print(f"  Fault {int(fault):<3} | Start Time: {int(start)}")

            print("\n📌 Fault Detected (Fault, End Index, Detection Time(s)):")
            for fault, end, detection in self.Fault_detected:
                print(f"  Fault {int(fault):<3} | End: {int(end)} | Detection: {int(detection)}s")

            print("\n📌 Average Detection Times (s):")

            for j in range(1,self.n):
                print(f"  Fault {j:<3} | Average Detection Time: {int(Ave_times[j])}s" if not np.isnan(Ave_times[j]) else "N/A")

            print(f"\n  Overall Average Detection Time: {int(np.nanmean(Ave_times[1:]))}s")
        
        self.Ave_time = Ave_times

    def detection_accuracy(self):

        for i in range(1, len(self.y_pred)):

            if self.y_test[i] > 0 and self.y_test[i] != self.y_test[i-1]:# self.y_test[i] != self.y_test[i-1]:
                self.fault = True
                self.counter += 1
                self.counteridx = np.append(self.counteridx, i)

            if self.fault and self.y_pred[i] == self.y_test[i] and self.y_test[i] != 0:
                self.correct += 1
                self.fault = False
                self.faultendidx = np.append(self.faultendidx, i)

            if self.fault and self.y_test[i] == 0:
                self.fault = False  # reset if fault clears without detection
                self.faultendidx = np.append(self.faultendidx, i)

            if self.y_pred[i] != self.y_test[i] and self.y_pred[i] != 0: #all
                if (i-self.FPidx)>300 and self.y_test[i+5] != self.y_pred[i] and self.y_test[i-5] != self.y_pred[i]:
                    self.FPcounter = self.FPcounter + 1 # count false positives
                    self.FPidx = i
                         
            if self.y_pred[i] != self.y_test[i] and self.y_test[i] == 0 and self.y_test[i-50] == 0: #no fault present
                if (i-self.FP0idx)>300 and self.y_test[i+5] != self.y_pred[i] and self.y_test[i-5] != self.y_pred[i]:
                    self.FP0counter = self.FP0counter + 1 # count false positives
                    self.FP0idx = i
                    self.indices = np.append(self.indices, i)
                    
        if self.counter == 0 and Display == 1:
            return "N/A (no fault transitions found in y_test)"
        else:

            print(f"Total Fault Transitions: {self.counter}")
            print(f"Detection Accuracy: {(self.correct/self.counter)*100:.0f}%")
            print(f"False Positives: {self.FPcounter}")
            print(f"False Positives (No Fault): {self.FP0counter}")
        
            return

# file = 'Classification_Results_PCA.h5'
# file = 'Classification_Results_Standard.h5'
# file = 'Classification_Results_Denoising.h5'
file = 'Classification_Results_UMAP.h5'

print('\n \n \n================= SVM ================\n \n \n')
y_test = load_vector(file, 'y_test_SVM')
y_pred = load_vector(file, 'y_pred_SVM')
counts = load_vector(file, 'counts')

SVM = metrics(y_test, y_pred, counts)
SVM.metrics()
SVM.detection_time()
SVM.detection_accuracy()
SVM.table()

if Display == 1:
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))  # use plt.subplots, not plt.figure
    ax.scatter(range(len(y_test)), y_test, label='True', s=30, alpha=0.6)
    ax.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', s=5, alpha=0.6)

    ax.legend()
    ax.set_title('SVM: True vs Predicted Faults')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Fault Class')
    ax.grid()
    plt.show()

print('\n \n \n ================= Random Forest ================ \n \n \n')
y_test = load_vector(file, 'y_test_RF')
y_pred = load_vector(file, 'y_pred_RF')
counts = load_vector(file, 'counts')

RF = metrics(y_test, y_pred, counts)
RF.metrics()
RF.detection_time()
RF.detection_accuracy()
RF.table()

if Display == 1:
# Plot
    fig, ax = plt.subplots(figsize=(12, 6))  # use plt.subplots, not plt.figure
    ax.scatter(range(len(y_test)), y_test, label='True', s=30, alpha=0.6)
    ax.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', s=5, alpha=0.6)

    ax.legend()
    ax.set_title('RF: True vs Predicted Faults')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Fault Class')
    ax.grid()
    plt.show()

print('\n \n \n ================= kNN ================ \n \n \n ')
y_test = load_vector(file, 'y_test_kNN')
y_pred = load_vector(file, 'y_pred_kNN')
counts = load_vector(file, 'counts')

kNN = metrics(y_test, y_pred, counts)
kNN.metrics()
kNN.detection_time()
kNN.detection_accuracy()
kNN.table()

if Display == 1:
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))  # use plt.subplots, not plt.figure
    ax.scatter(range(len(y_test)), y_test, label='True', s=30, alpha=0.6)
    ax.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', s=5, alpha=0.6)

    ax.legend()
    ax.set_title('kNN: True vs Predicted Faults')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Fault Class')
    ax.grid()
    plt.show()
