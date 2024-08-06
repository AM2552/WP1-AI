from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras.applications import EfficientNetB1
from keras import backend, layers
from keras.regularizers import l2
from dataset_generation_birds import train_dataset, validation_dataset
import matplotlib.pyplot as plt
from PIL import Image

def lr_schedule(epoch):
    if epoch > 40:
        return 0.0001
    elif epoch > 30:
        return 0.0005
    elif epoch > 15:
        return 0.005
    else:
        return 0.01

def train_model(initial_learning_rate, optimizer_name, epochs, preset_name=''):
    inputs = layers.Input(shape=(240, 240, 3))
    model = EfficientNetB1(include_top=False, weights='imagenet', input_tensor=inputs)
    model.trainable = False
    
    x = layers.GlobalAveragePooling2D()(model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(200, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.summary()
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=initial_learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=initial_learning_rate, momentum=0.9, nesterov=True)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    

    #lr_scheduler = LearningRateScheduler(lr_schedule)

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        #callbacks=[lr_scheduler],
        shuffle=True
    )
    
    train_accuracy_history = history.history['accuracy']
    val_accuracy_history = history.history['val_accuracy']
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    model.save(f'birds_eff1_{preset_name}.h5')
    
    return train_accuracy_history, val_accuracy_history, train_loss_history, val_loss_history

results = []

parameter_presets = {
    'adam': (0.001, 'adam', 30),
    #'adam_wtop_30': (0.001, 'adam', 30),
    #'adam2': (0.0007, 'adam', 50),
    #'adam3': (0.0005, 'adam', 50),
    #'adam4': (0.0003, 'adam', 50),
}

for preset_name, parameters in parameter_presets.items():
    backend.clear_session()
    train_accuracy_history , val_accuracy_history, train_loss_history , val_loss_history = train_model(*parameters, preset_name)
    results.append((val_accuracy_history[-1], preset_name))
    
    plt.plot(val_accuracy_history, label=f'{preset_name} - LR({parameters[0]})')

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    # First subplot for accuracy
    axs[0].plot(train_accuracy_history, label='Train Accuracy')
    axs[0].plot(val_accuracy_history, label='Validation Accuracy')
    axs[0].set_title(f'{preset_name} - LR({parameters[0]}) ')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    
    # Second subplot for loss
    axs[1].plot(train_loss_history, label='Train Loss')
    axs[1].plot(val_loss_history, label='Validation Loss')
    axs[1].set_title(f'{preset_name} - LR({parameters[0]}) ')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'birds_eff1_{preset_name}.png')
    
results.sort(reverse=True)
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')

#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend()
#plt.savefig('bcd_sgd_test.png')
#plt.show()
