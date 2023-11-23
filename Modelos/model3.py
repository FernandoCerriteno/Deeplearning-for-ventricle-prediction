import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization, Activation, ReLU, Conv2DTranspose

filters = [64, 128, 256, 512]

# Definición de la arquitectura U-Net
def unet_model(input_shape=(112, 112, 1), n_classes=1, activation='sigmoid'):
    # Entrada de la red
    inputs = Input(shape=input_shape)
    previous_layer = inputs


    contract_layers = {}
    # Capas de contracción
    for i in filters:
        conv = Conv2D(i, (3,3), activation = 'relu',kernel_initializer='he_normal', padding='same')(previous_layer)
        conv = Dropout(0.1)(conv)
        conv = Conv2D(i, (3,3), activation = 'relu',kernel_initializer='he_normal', padding='same')(conv)
        contract_layers[f'conv{i}'] = conv
        conv = MaxPooling2D(pool_size=(2, 2))(conv)
        previous_layer = conv

    cx = Conv2D(1024, (3,3), activation = 'relu',kernel_initializer='he_normal', padding='same')(previous_layer)
    cx = Dropout(0.2)(cx)
    cx = Conv2D(1024, (3,3), activation = 'relu',kernel_initializer='he_normal', padding='same')(cx)
    previous_layer = cx

    # Capas de expansión
    for i in reversed(filters):
        up = Conv2DTranspose(i, (2,2), strides=(2,2), padding='same')(previous_layer)
        up = concatenate([up, contract_layers[f'conv{i}']])
        conv = Conv2D(i, (3,3), activation = 'relu',kernel_initializer='he_normal', padding='same')(up)
        conv = Dropout(0.1)(conv)
        conv = Conv2D(i, (3,3), activation = 'relu',kernel_initializer='he_normal', padding='same')(conv)
        previous_layer = conv

    outputs = Conv2D(n_classes, (1,1), activation = activation)(previous_layer)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='U-Net_3')
    
    return model