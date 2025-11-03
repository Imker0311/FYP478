# %%
import umap
from umap.umap_ import UMAP
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

# Load training data
file = 'Dataframe.h5'
X = load_vector(file, 'X')
Y = load_vector(file, 'Y')

# %%
def build_umap(
    n_components,           # Size of the embedding (like encoding_dim)
    n_neighbors,            # Local neighborhood size
    min_dist,               # Controls tightness of clusters
    metric                  # Distance metric
):

    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric
    )
    return reducer

# Build UMAP model
umap_model = build_umap(
    n_components=5,    # same as encoding_dim in AE
    n_neighbors=200,
    min_dist=0.2,
    metric='euclidean'
)

# Train UMAP to training data
X_train = umap_model.fit_transform(X)
print("UMAP embedding shape:", X_train.shape)

# Save training data
file = 'DimRed_UMAP.h5'
save_vector(file, 'X_train', X_train)
save_vector(file, 'y_train', Y)

# Load test data
file = 'Dataframe.h5'
X_test = load_vector(file, 'X_test')
y_test = load_vector(file, 'y_test')
counts = load_vector(file, 'counts')

# Apply UMAP to test data
X_compressed = umap_model.transform(X_test)
X_test = X_compressed
print("UMAP embedding shape:", X_test.shape)

# Save test data
file = 'DimRed_UMAP.h5'
save_vector(file, 'X_test', X_test)
save_vector(file, 'y_test', y_test)

file = 'Classification_Results_UMAP.h5'
save_vector(file, 'counts', counts)
