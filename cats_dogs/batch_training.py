from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from dataset_generation import train_generator, validation_generator
import matplotlib.pyplot as plt

def train_model(conv_layers, dense_layers, learning_rate, epochs, dropout=bool):
    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    filters = 64
    for _ in range(conv_layers - 1):
        model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        filters *= 2  # Double the number of filters for the next layer

    model.add(Flatten())
    
    for _ in range(dense_layers):
        model.add(Dense(64, activation='relu'))
    
    if dropout:
        model.add(Dropout(0.2))  # Add dropout to prevent overfitting
    
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    optimizer = Adam(learning_rate=learning_rate)

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
    accuracy_history = history.history['val_accuracy']
    #################
    model.save('cats_vs_dogs_model.h5')
    #################
    # Return the final validation accuracy
    return accuracy_history

# List to store results
results = []

# Parameter combinations to try
parameter_presets = {
    #'Preset1': (5, 3, 0.0005, 50, True),
    #'Preset2': (5, 3, 0.00025, 50, True),
    #'Preset3': (5, 3, 0.00025, 50, False),
    #'Preset4': (6, 3, 0.0005, 50, True),
    #'Preset5': (6, 1, 0.00025, 80, False), 
    'Preset6': (6, 2, 0.00025, 80, True), # Best preset  
}

# Train the model with each combination of parameters
for preset_name, parameters in parameter_presets.items():
    accuracy_history = train_model(*parameters)
    results.append((accuracy_history[-1], preset_name))
    
    # Plot the accuracy history
    plt.plot(accuracy_history, label=preset_name)
    

# Sort the results by accuracy
results.sort(reverse=True)

# Print the sorted results
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('cat&dog_accuracy.png')
plt.show()