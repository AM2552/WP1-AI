from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from dataset_generation import train_generator, validation_generator
import matplotlib.pyplot as plt

def train_model(conv_layers, dense_layers, optimizer, epochs, dropout=bool):
    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))  # Add dropout to prevent overfitting
    
    filters = 64
    for _ in range(conv_layers - 1):
        model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dropout:
            model.add(Dropout(0.25))  # Add dropout to prevent overfitting
        filters *= 2  # Double the number of filters for the next layer

    model.add(Flatten())
    
    for _ in range(dense_layers):
        model.add(Dense(64, activation='relu'))
        if dropout:
            model.add(Dropout(0.25))  # Add dropout to prevent overfitting
    
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=50
    )
    
    # Return the final validation accuracy
    return history.history['val_accuracy']

# List to store results
results = []

# Parameter combinations to try
parameter_presets = {
    'Preset1': (2, 1, 'adam', 25, False),
    'Preset2': (2, 1, 'adam', 25, True),
    'Preset3': (3, 1, 'adam', 25, False),
    'Preset4': (3, 1, 'adam', 25, True),
    'Preset5': (3, 2, 'adam', 25, False),
    'Preset6': (3, 2, 'adam', 25, True),
    'Preset7': (2, 1, 'sgd', 25, False),
    'Preset8': (2, 1, 'sgd', 25, True),
    'Preset9': (3, 1, 'sgd', 25, False),
    'Preset10': (3, 1, 'sgd', 25, True),
    'Preset11': (3, 2, 'sgd', 25, False),
    'Preset12': (3, 2, 'sgd', 25, True),
}

# Train the model with each combination of parameters
for preset_name, params in parameter_presets.items():
    accuracy_history = train_model(*params)
    single_accuracy = accuracy_history[-1]
    results.append((single_accuracy, preset_name))  # Store the preset name instead of the params
    
    # Plot the accuracy history
    plt.plot(accuracy_history, label=preset_name)  # Use the preset name as the label

# Sort the results by accuracy
results.sort(reverse=True)

# Print the sorted results
for accuracy, params in results:
    print(f'Accuracy: {accuracy:.4f}, Params: {params}')
    
plt.legend()
plt.show()