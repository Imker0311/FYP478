# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tensorflow import keras
from keras import layers, Model, optimizers, Input
import sys
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--Hidden_Layers", type=str, required=True, help="Hidden layer sizes (Python list as string)")
args = parser.parse_args()

# Convert string "[20, 15, 10]" → Python list [20, 15, 10]
Hidden_Layers = ast.literal_eval(args.Hidden_Layers)

def load_vector(filename, name):
    with h5py.File(filename, 'r') as f:
        data = f[name][:]
        #print(f"Loaded dataset '{name}' with shape {data.shape}")
        return data

def save_vector(filename, name, data): # Creates dataset if it does not exist, overwrites if it does
    with h5py.File(filename, 'a') as f:  # open file in append mode
        if name in f:
            del f[name]  # delete old dataset before overwriting
        f.create_dataset(name, data=data, chunks=True, compression='gzip')
        #print(f"Saved dataset '{name}' with shape {data.shape}")

file = 'Dataframe.h5'
X = load_vector(file, 'X')
Y = load_vector(file, 'Y')

X_train = X

# params = sys.argv[1:]

# %% [markdown]
# Standard Autoencoder

# %%
def build_autoencoder(
    input_dim,
    encoding_dim,
    hidden_layers,  # List of units for hidden layers before bottleneck
    activation,
    bottleneck_activation,
    output_activation,
    optimizer='adam',
    learning_rate=0.001,
    loss='mse'
):
    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Encoder
    x = input_layer
    for units in hidden_layers:
        x = layers.Dense(units, activation=activation)(x)
    encoded = layers.Dense(encoding_dim, activation=bottleneck_activation, name='encoding_layer')(x)

    # Decoder (mirror of encoder)
    x = encoded
    for units in reversed(hidden_layers):
        x = layers.Dense(units, activation=activation)(x)
    output_layer = layers.Dense(input_dim, activation=output_activation)(x)

    # Autoencoder model: input to reconstructed output
    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    # Encoder model: input to bottleneck representation
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Set optimizer
    opt_lower = optimizer.lower()
    if opt_lower == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif opt_lower == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate)
    elif opt_lower == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    elif opt_lower == 'adagrad':
        opt = optimizers.Adagrad(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer selected.")

    # Compile autoencoder
    autoencoder.compile(optimizer=opt, loss=loss)

    # Return both models
    return autoencoder, encoder


# Build autoencoder model
autoencoder, encoder = build_autoencoder(
    input_dim=14,
    encoding_dim=5,
    hidden_layers=Hidden_Layers,  # Example hidden layers
    activation='tanh',
    bottleneck_activation='linear',
    output_activation='linear',
    optimizer='adam',
    learning_rate=0.001,
    loss='mse')
encoder.summary()

# Autoencoder training
history = autoencoder.fit(
    X_train,
    X_train,  # Autoencoders use input as output
    epochs=25,
    batch_size=32,
    #validation_data=(X_test, X_test),
    validation_data=(X_train, X_train),
    verbose=0)

val_loss = history.history['val_loss'][-1]
print("Val Loss:",val_loss)

# %%
X_compressed = encoder.predict(X)

X_train = X_compressed
y_train = Y

# Save reduced data for classifier training

file = 'DimRed_AE_Standard.h5'
save_vector(file, 'X_train', X_train)
save_vector(file, 'y_train', y_train)

# %%
X_recon = autoencoder.predict(X)

# plt.figure(figsize=(12, 6))
# plt.plot(X_recon[130000:150000, 8], label='Reconstructed', color='orange',linewidth=1.5)
# plt.plot(X[130000:150000,8], label='Original Signal', alpha=0.5,linewidth=1.5)
# plt.title('AutoEncoder Reconstructed Signal')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.legend()
# plt.show() 

# %% [markdown]
# Test Dataset

# %%
file = 'Dataframe.h5'
X_test = load_vector(file, 'X_test')
y_test = load_vector(file, 'y_test')
counts = load_vector(file, 'counts')

# %%
X_test_plot = X_test

X_compressed = encoder.predict(X_test)
X_test = X_compressed

X_recon = autoencoder.predict(X_test_plot)

file = 'DimRed_AE_Standard.h5'

save_vector(file, 'X_test', X_test)
save_vector(file, 'y_test', y_test)

file = 'Classification_Results_Standard.h5'
save_vector(file, 'counts', counts)

# # %%
# plt.figure(figsize=(12, 6))
# plt.plot(X_recon[:, 2], label='Reconstructed', color='orange',linewidth=1.5)
# plt.plot(X_test_plot[:,2], label='Original Signal', alpha=0.5,linewidth=1.5)
# plt.title('AutoEncoder Reconstructed Signal')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# # 3D plot

# idx = np.where(Y!=0)  # indices of non-zero elements

# fig = plt.figure(figsize=(15, 6))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(X_train[idx, 0], X_train[idx, 1], X_train[idx, 2],c=Y[idx],cmap='viridis',alpha=1.0,s=20,edgecolors='none')
# cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
# cbar.set_label('Fault Class')
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# plt.show()


