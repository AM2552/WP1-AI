from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.applications import EfficientNetB0, EfficientNetB1
from keras import backend
from dataset_generation_bcd import train_generator, validation_generator
import matplotlib.pyplot as plt


def train_model(dense_layers, learning_rate, optimizer, epochs, dropout=bool, preset_name=str()):
    base_model = EfficientNetB1(include_top=False, weights=None, input_shape=(256, 256, 3))
    base_model.trainable = True  # Train the base model from scratch
    
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    
    neurons = 512
    for _ in range(dense_layers):
        model.add(Dense(neurons))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        neurons /= 2
        if dropout:
            model.add(Dropout(0.3))
    
    model.add(Dense(neurons))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))

    model.summary()
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
    )
    
    accuracy_history = history.history['val_accuracy']
    model.save(f'bird_cat_dog_model_{preset_name}_eff1.h5')
    
    return accuracy_history

results = []

parameter_presets = {
    'adam': (2, 0.0001, 'adam', 50, True),
    'sgd': (2, 0.005, 'sgd', 50, True),
    #'Preset3': (3, 0.0001, 'adam', 50, True),
}

for preset_name, parameters in parameter_presets.items():
    backend.clear_session()
    accuracy_history = train_model(*parameters, preset_name)
    results.append((accuracy_history[-1], preset_name))
    
    plt.plot(accuracy_history, label=f'{preset_name} - DL({parameters[0]})')
    
results.sort(reverse=True)
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('bird_cat_dog_accuracy_eff1.png')
plt.show()
