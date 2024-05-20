from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, SGD
from keras import backend
from dataset_generation import train_generator, validation_generator
import matplotlib.pyplot as plt


def conv_layer(model, filters):
    model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

def conv_block(model, filters, layers):
    for _ in range(layers):
        conv_layer(model, filters)
    model.add(MaxPooling2D(pool_size=(2, 2)))

def train_model(conv_layers, dense_layers, learning_rate, optimizer, epochs, dropout=bool,  preset_name=str()):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    filters = 64
    for _ in range(conv_layers - 1):
        conv_layer(model, filters)
        filters *= 2
    
    model.add(Flatten())
    neurons = 128
    for _ in range(dense_layers):
        model.add(Dense(neurons, activation='relu'))
        neurons /= 2
    if dropout:
        model.add(Dropout(0.3))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.summary()
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=50
    )
    
    accuracy_history = history.history['val_accuracy']
    model.save(f'bird_cat_dog_model_{preset_name}.h5')
    
    return accuracy_history

results = []

parameter_presets = {
    'Preset1': (6, 3, 0.0001, 'adam', 200, True),
    #'Preset2': (6, 3, 0.00005, 'adam', 100, True),
    #'Preset3': (6, 3, 0.00001, 'adam', 100, True),
    #'Preset4': (6, 3, 0.0001, 'sgd', 100, True),
    #'Preset5': (6, 3, 0.00005, 'sgd', 100, True),
    #'Preset6': (6, 3, 0.00001, 'sgd', 100, True),
    #'Preset7': (6, 0, 0.0001, 50, False),
    #'Preset8': (6, 0, 0.00001, 50, False),
}

for preset_name, parameters in parameter_presets.items():
    backend.clear_session()
    accuracy_history = train_model(*parameters, preset_name)
    results.append((accuracy_history[-1], preset_name))
    
    plt.plot(accuracy_history, label=f'{preset_name} - CL({parameters[0]}) DL({parameters[1]}) LR({parameters[2]}) {parameters[3]}')
    
results.sort(reverse=True)
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('bird_cat_dog_accuracy.png')
plt.show()