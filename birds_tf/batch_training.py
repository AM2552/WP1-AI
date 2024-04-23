from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
from dataset_generation import train_generator, validation_generator
import matplotlib.pyplot as plt

def train_model(conv_layers, dense_layers, learning_rate, epochs, dropout=True, preset_name=""):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), use_bias=False, input_shape=(256, 256, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    filters = 64
    for _ in range(conv_layers - 1):
        model.add(Conv2D(filters, (3, 3), use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        filters *= 2
    model.add(Flatten())
    for _ in range(dense_layers):
        model.add(Dense(64, activation='relu'))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Dense(200, activation='softmax'))  # Updated number of outputs and activation function

    model.summary()
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  # Updated loss function

    history = model.fit(
        train_generator,
        steps_per_epoch=100,  # Adjust these steps based on your actual data and batch size
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=50  # Adjust these steps as well
    )
    
    accuracy_history = history.history['val_accuracy']
    model.save(f'bird_classification_model_{preset_name}.h5')  # Updated model naming
    
    return accuracy_history

results = []

parameter_presets = {
    'Preset1': (6, 4, 0.0001, 30, True),
    # Additional presets can be defined here
}

for preset_name, parameters in parameter_presets.items():
    backend.clear_session()
    accuracy_history = train_model(*parameters, preset_name)
    results.append((accuracy_history[-1], preset_name))
    
    plt.plot(accuracy_history, label=preset_name)
    
results.sort(reverse=True)
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('bird_classification_accuracy.png')
plt.show()
