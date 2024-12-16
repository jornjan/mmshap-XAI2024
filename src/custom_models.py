from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import resnet



def StaticModel(num_features) -> Model:
    inputs = layers.Input(shape=(num_features,))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal', name='feature_layer')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs)


def AugmentationLayer() -> Model:
    inputs = layers.Input(shape=(224,224,3))
    x = preprocessing.Rescaling(scale=1./255)(inputs)
    x = preprocessing.RandomFlip()(x)
    x = preprocessing.RandomRotation(20/360)(x)
    x = preprocessing.RandomTranslation(0.2, 0.2)(x)
    outputs = x
    return Model(inputs, outputs, name='augmentation')
    

def ImageModel() -> Model:
    inputs = layers.Input(shape=(224,224,3))
    aug_layer = AugmentationLayer()(inputs)
    
    basenet = resnet.ResNet50(input_shape=(224,224,3),
                              weights='imagenet',
                              include_top=False,
                              pooling='max')
    
    x = basenet(aug_layer)
    x = layers.Dense(256, activation=layers.LeakyReLU(), kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal', name='feature_layer')(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)


def VitalsModel() -> Model:
    inputs = layers.Input(shape=(None, 6))
    x = layers.Masking(mask_value=-10.0)(inputs)
    lstm_layer = layers.LSTM(128, dropout=0.5)
    x = layers.Bidirectional(lstm_layer)(x)
    x = layers.Dense(128, activation=layers.LeakyReLU(), kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal', name='feature_layer')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)


def MedicationModel() -> Model:
    inputs = layers.Input(shape=(17,), name='feature_layer')
    outputs = layers.Dense(1, activation='sigmoid')(inputs)

    return Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    static_model = StaticModel(5)
    static_model.summary(150)

    image_model = ImageModel()
    image_model.summary(150)
    
    vitals_model = VitalsModel()
    vitals_model.summary(150)
    
    med_model = MedicationModel()
    med_model.summary(150)