# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py

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

# Load training Data
file = 'Dataframe.h5'
X = load_vector(file, 'X')
Y = load_vector(file, 'Y')

# %%
from sklearn.decomposition import PCA

# Create scree plot
pca = PCA()
df_pca = pca.fit_transform(X)

exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

plt.figure(figsize=(10, 6))
plt.step(range(1, len(cum_var) + 1), cum_var, where='mid', label='Cumulative explained variance', color='black')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.legend()
plt.grid(True)

# %%
# Train PCA
pca = PCA(n_components=5)
X_train = pca.fit_transform(X)
y_train = Y

# Save training data
file = 'DimRed_PCA.h5'
save_vector(file, 'X_train', X_train)
save_vector(file, 'y_train', y_train)

# %%
# Load test data
file = 'Dataframe.h5'
X_test = load_vector(file, 'X_test')
y_test = load_vector(file, 'y_test')
counts = load_vector(file, 'counts')

# Transform test data
X_test = pca.transform(X_test)

# Save test data
file = 'DimRed_PCA.h5'
save_vector(file, 'X_test', X_test)
save_vector(file, 'y_test', y_test)

# Saves fault counts to 
file = 'Classification_Results_PCA.h5'
save_vector(file, 'counts', counts)

# %%
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y, cmap='viridis', alpha=0.5, s=10)
plt.colorbar(label='Fault Class')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of CSTR Simulation Data')

#3D plot
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train[:, 1], X_train[:, 2], X_train[:, 0],c=Y, cmap='viridis', alpha=0.5, s=10)
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Fault Class')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')


