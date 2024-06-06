import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, backend as K
import numpy as np
from bird_cat_dog.dataset_generation_bcd import train_generator, validation_generator
import matplotlib.pyplot as plt

def conv_layer(x, filters):
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x

def rpn_layer(base_layers, num_anchors):
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = layers.Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = layers.Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    return [x_class, x_regr, base_layers]

class ROIPoolingLayer(layers.Layer):
    def __init__(self, pool_size, **kwargs):
        self.pool_size = pool_size
        super(ROIPoolingLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return None, self.pool_size, self.pool_size, input_shape[0][3]

    def call(self, inputs):
        feature_map, rois = inputs
        outputs = []
        for roi in rois:
            x, y, w, h = roi[1:]
            x, y, w, h = int(x), int(y), int(w), int(h)
            region = feature_map[:, y:y+h, x:x+w, :]
            pooled_region = tf.image.resize(region, (self.pool_size, self.pool_size))
            outputs.append(pooled_region)
        return tf.concat(outputs, axis=0)

def build_model(input_shape=(256, 256, 3), conv_layers=4, dense_layers=2, learning_rate=0.0001, optimizer='adam', dropout=True):
    input_tensor = layers.Input(shape=input_shape)
    roi_input = layers.Input(shape=(None, 4))

    # Backbone CNN
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    filters = 64
    for _ in range(conv_layers - 1):
        x = conv_layer(x, filters)
        filters *= 2

    # RPN
    num_anchors = 9  # This can be changed based on your anchor box configuration
    rpn = rpn_layer(x, num_anchors)

    # ROI Pooling
    pooled_regions = ROIPoolingLayer(pool_size=7)([rpn[2], roi_input])

    # Classification and Regression heads
    x = layers.TimeDistributed(layers.Flatten())(pooled_regions)
    x = layers.TimeDistributed(layers.Dense(4096, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(4096, activation='relu'))(x)

    cls_output = layers.TimeDistributed(layers.Dense(3, activation='softmax'))(x)
    regr_output = layers.TimeDistributed(layers.Dense(3 * 4, activation='linear'))(x)

    model = models.Model([input_tensor, roi_input], [cls_output, regr_output, rpn[0], rpn[1]])

    # Compile the model
    if optimizer == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, 
                  loss={'time_distributed_2': 'categorical_crossentropy', 
                        'time_distributed_3': 'mse', 
                        'rpn_out_class': 'binary_crossentropy', 
                        'rpn_out_regress': 'mse'}, 
                  metrics={'time_distributed_2': 'accuracy', 
                           'rpn_out_class': 'accuracy'})

    model.summary()
    return model

def train_model(model, epochs):
    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=50
    )
    
    return history

# Parameters
conv_layers = 4
dense_layers = 2
learning_rate = 0.0001
optimizer = 'adam'
epochs = 200
dropout = True
input_shape = (256, 256, 3)

# Build and train the model
model = build_model(input_shape, conv_layers, dense_layers, learning_rate, optimizer, dropout)
history = train_model(model, epochs)

# Plot the accuracy
plt.plot(history.history['val_time_distributed_2_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('bird_cat_dog_accuracy.png')
plt.show()
