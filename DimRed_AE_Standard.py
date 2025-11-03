# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tensorflow import keras
from keras import layers, Model, optimizers, Input

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

# Autoencoder structure
def build_autoencoder(
    input_dim,
    encoding_dim,
    hidden_layers,  # List of units for hidden layers before bottleneck
    activation,
    bottleneck_activation,
    output_activation,
    optimizer,
    learning_rate,
    loss
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
    hidden_layers=[20, 14, 10],  # Example hidden layers
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
# Load test data
file = 'Dataframe.h5'
X_test = load_vector(file, 'X_test')
y_test = load_vector(file, 'y_test')
counts = load_vector(file, 'counts')

# %%
# Apply AE to test data
X_compressed = encoder.predict(X_test)
X_test = X_compressed

# Save test data
file = 'DimRed_AE_Standard.h5'
save_vector(file, 'X_test', X_test)
save_vector(file, 'y_test', y_test)

file = 'Classification_Results_Standard.h5'
save_vector(file, 'counts', counts)

