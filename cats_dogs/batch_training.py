from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras import backend
from dataset_generation import train_generator, validation_generator
import matplotlib.pyplot as plt


def conv_layer(model, filters):
    model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

def conv_layer_batch_norm(model, filters):
    model.add(Conv2D(filters, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

def conv_block(model, filters, layers):
    for _ in range(layers):
        conv_layer(model, filters)
    model.add(MaxPooling2D(pool_size=(2, 2)))

def train_model(conv_layers, dense_layers, learning_rate, optimizer, epochs, dropout=bool, batch=bool,  preset_name=str()):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    filters = 64
    if not batch:
        for _ in range(conv_layers - 1):
            conv_layer(model, filters)
            filters *= 1.5
    else:
        for _ in range(conv_layers - 1):
            conv_layer_batch_norm(model, filters)
            filters *= 1.5
    
    model.add(GlobalAveragePooling2D())
    neurons = 512
    for _ in range(dense_layers):
        if not batch:
            model.add(Dense(neurons, activation='relu'))
        else:
            model.add(Dense(neurons))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        neurons /= 2
        if dropout:
            model.add(Dropout(0.3))
    if not batch:
        model.add(Dense(neurons, activation='relu'))
    else:
        model.add(Dense(neurons))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
    )
    
    train_accuracy_history = history.history['accuracy']
    val_accuracy_history = history.history['val_accuracy']
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']
    model.save(f'cats_vs_dogs_model_{preset_name}.h5')
    
    return train_accuracy_history, val_accuracy_history, train_loss_history, val_loss_history


results = []

parameter_presets = {
    'Preset1': (6, 3, 0.0001, 'adam', 50, True, False),
    'Preset2': (6, 3, 0.0001, 'adam', 50, True, True),
    #'Preset3': (6, 3, 0.00001, 'adam', 100, True),
    #'Preset4': (6, 3, 0.0001, 'sgd', 100, True),
    #'Preset5': (6, 3, 0.00005, 'sgd', 100, True),
    #'Preset6': (6, 3, 0.00001, 'sgd', 100, True),
    #'Preset7': (6, 0, 0.0001, 50, False),
    #'Preset8': (6, 0, 0.00001, 50, False),
}

for preset_name, parameters in parameter_presets.items():
    backend.clear_session()
    train_accuracy_history , val_accuracy_history, train_loss_history , val_loss_history = train_model(*parameters, preset_name)
    results.append((val_accuracy_history[-1], preset_name))
    
    plt.plot(val_accuracy_history, label=f'{preset_name} - CL({parameters[0]}) DL({parameters[1]}) LR({parameters[2]}) {parameters[3]}')
    
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    # First subplot for accuracy
    axs[0].plot(train_accuracy_history, label='Train Accuracy')
    axs[0].plot(val_accuracy_history, label='Validation Accuracy')
    axs[0].set_title(f'{preset_name} - CL({parameters[0]}) DL({parameters[1]}) LR({parameters[2]}) {parameters[3]}')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    
    # Second subplot for loss
    axs[1].plot(train_loss_history, label='Train Loss')
    axs[1].plot(val_loss_history, label='Validation Loss')
    axs[1].set_title(f'{preset_name} - CL({parameters[0]}) DL({parameters[1]}) LR({parameters[2]}) {parameters[3]}')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'cat_vs_dog_{preset_name}.png')

results.sort(reverse=True)
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')

#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend()
#plt.savefig('cat&dog_accuracy.png')
#plt.show()