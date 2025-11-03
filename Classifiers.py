import h5py
import numpy as np

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

# file = 'DimRed_PCA.h5'
# file = 'DimRed_AE_Standard.h5'
# file = 'DimRed_AE_Denoising.h5'
file = 'DimRed_UMAP.h5'

X_test = load_vector(file, 'X_test')
y_test = load_vector(file, 'y_test')
X_train = load_vector(file, 'X_train')
y_train = load_vector(file, 'y_train')

# file = 'Classification_Results_PCA.h5'
# file = 'Classification_Results_Standard.h5'
# file = 'Classification_Results_Denoising.h5'
file = 'Classification_Results_UMAP.h5'

# kSVM
from sklearn import svm

SVM = svm.SVC(
    kernel='rbf', 
    C=1, 
    gamma='scale',
    probability=True
)

SVM.fit(X_train, y_train)

# --- Probabilities ---
probs = SVM.predict_proba(X_test)           # shape (n_samples, n_classes)
max_probs = np.max(probs, axis=1)           # highest probability per sample
preds = np.argmax(probs, axis=1)            # class with highest probability

# --- Apply threshold ---
threshold = 0.9
y_pred = np.where(max_probs >= threshold, preds, 0)   # 0 = "no fault"

save_vector(file, 'y_pred_SVM', y_pred)
save_vector(file, 'y_test_SVM', y_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
)
rf.fit(X_train, y_train)

# --- Probabilities ---
probs = rf.predict_proba(X_test)           # shape (n_samples, n_classes)
max_probs = np.max(probs, axis=1)          # highest probability per sample
preds = np.argmax(probs, axis=1)           # class with highest probability

# --- Apply threshold ---
threshold = 0.9
y_pred = np.where(max_probs >= threshold, preds, 0)   # 0 = "no fault"

save_vector(file, 'y_pred_RF', y_pred)
save_vector(file, 'y_test_RF', y_test)

# k Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=200,
    weights="distance",
    metric="minkowski",
    p=2
)

knn.fit(X_train, y_train)

# --- Probabilities ---
probs = knn.predict_proba(X_test)          # shape (n_samples, n_classes)
max_probs = np.max(probs, axis=1)          # highest probability per sample
preds = np.argmax(probs, axis=1)           # class with highest probability

# --- Apply threshold ---
threshold = 0.9
y_pred = np.where(max_probs >= threshold, preds, 0)   # 0 = "no fault"


save_vector(file, 'y_pred_kNN', y_pred)
save_vector(file, 'y_test_kNN', y_test)