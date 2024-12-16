import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics, optimizers, callbac

import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

import custom_models, custom_generators
from util_functions import *

from typing import Tuple, Dict



############## STATIC MODEL ####################

def impute_static_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k_neighbors_regressor = KNeighborsRegressor(n_neighbors=10)
    imputer = IterativeImputer(estimator=k_neighbors_regressor,
                                    max_iter=20, verbose=0)
    
    X_train = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val = pd.DataFrame(imputer.transform(X_val), index=X_val.index, columns=X_val.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train, X_val, X_test


def normalize_static_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    high_cols = (X_train>1).any(axis=0)
    high_cols = high_cols[high_cols].index.tolist()
    
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train, X_val, X_test



def train_static_model(target_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, keras.Model]:
    fname = 'PATH/TO/PATIENT/DATA.pkl'
    raise NotImplementedError('Please add the local path to static data')
    static_df = pd.read_pickle(fname)

    # Drop target columns
    static_df = static_df.drop(columns=['overl30d', 'C_ANEM', 'C_DC', 'C_PNEU', 'C_DEL'])

    y_train = target_dict['train'].copy()
    y_val = target_dict['val'].copy()
    y_test = target_dict['test'].copy()
    
    X_train = static_df.loc[y_train.index.tolist()]
    X_val = static_df.loc[y_val.index.tolist()]
    X_test = static_df.loc[y_test.index.tolist()]
    
    X_train, X_val, X_test = impute_static_data(X_train, X_val, X_test)

    # Save imputed data
    pd.concat([X_train, X_val, X_test]).to_pickle('./temp_imputed_static.pkl')

    model = custom_models.StaticModel(len(X_train.columns))
    model.compile(loss=custom_binary_entropy_loss(y_train), optimizer=optimizers.Adam(1e-3), metrics=get_unimodal_metrics(), run_eagerly=True)
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks(),
        verbose=0
    )
    
    train_pred = model.predict(X_train, batch_size=32, verbose=0)
    val_pred = model.predict(X_val, batch_size=32, verbose=0)
    test_pred = model.predict(X_test, batch_size=32, verbose=0)
    results_df = compute_metrics([y_train, y_val, y_test],
                                 [train_pred, val_pred, test_pred])
    results_df.insert(0, 'model', 'static')
    return results_df, model
    


############## IMAGE MODEL ####################

def train_image_model(target_dict: Dict[str, pd.DataFrame], 
                      modality: str) -> Tuple[pd.DataFrame, keras.Model]:
    
    train_gen = custom_generators.ImageDataGenerator(target_dict['train'].copy(), modality)
    val_gen = custom_generators.ImageDataGenerator(target_dict['val'].copy(), modality, shuffle=False)
    test_gen = custom_generators.ImageDataGenerator(target_dict['test'].copy(), modality, shuffle=False)

    model = custom_models.ImageModel()
    model.compile(loss=custom_binary_entropy_loss(target_dict['train'].copy()), optimizer=optimizers.Adam(1e-5), metrics=get_unimodal_metrics())
    model.fit(
        train_gen,
        epochs=100,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=0
    )
    
    train_gen.shuffle = False
    train_pred = model.predict(train_gen, batch_size=32, verbose=0)
    val_pred = model.predict(val_gen, batch_size=32, verbose=0)
    test_pred = model.predict(test_gen, batch_size=32, verbose=0)
    results_df = compute_metrics([train_gen.get_labels(), val_gen.get_labels(), test_gen.get_labels()],
                                 [train_pred, val_pred, test_pred])
    results_df.insert(0, 'model', modality)
    return results_df, model


############## VITALS MODEL ####################

def train_vitals_model(target_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, keras.Model]:
    train_gen = custom_generators.VitalsGenerator(target_dict['train'].copy())
    val_gen = custom_generators.VitalsGenerator(target_dict['val'].copy())
    test_gen = custom_generators.VitalsGenerator(target_dict['test'].copy())
    
    model = custom_models.VitalsModel()
    model.compile(loss=custom_binary_entropy_loss(target_dict['train'].copy()), optimizer=optimizers.Adam(5e-4), metrics=get_unimodal_metrics())
    model.fit(
        train_gen,
        epochs=100,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=0
    )
    
    train_gen.shuffle = False
    train_pred = model.predict(train_gen, batch_size=32, verbose=0)
    val_pred = model.predict(val_gen, batch_size=32, verbose=0)
    test_pred = model.predict(test_gen, batch_size=32, verbose=0)
    results_df = compute_metrics([train_gen.get_labels(), val_gen.get_labels(), test_gen.get_labels()],
                                 [train_pred, val_pred, test_pred])
    results_df.insert(0, 'model', 'vitals')
    return results_df, model



############## MEDICATION MODEL ####################
def train_med_model(target_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, keras.Model]:
    fname = 'PATH/TO/PATIENT/DATA.pkl'
    raise NotImplementedError('Please add the local path to medication data')
    
    y_train = target_dict['train'].copy()
    y_val = target_dict['val'].copy()
    y_test = target_dict['test'].copy()
    
    X_train = medication_df.loc[y_train.index.tolist()]
    X_val = medication_df.loc[y_val.index.tolist()]
    X_test = medication_df.loc[y_test.index.tolist()]
    
    model = custom_models.MedicationModel()
    model.compile(loss=custom_binary_entropy_loss(y_train), optimizer=optimizers.Adam(1e-4), metrics=get_unimodal_metrics())
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks(),
        verbose=0
    )
    
    train_pred = model.predict(X_train, batch_size=32, verbose=0)
    val_pred = model.predict(X_val, batch_size=32, verbose=0)
    test_pred = model.predict(X_test, batch_size=32, verbose=0)
    results_df = compute_metrics([y_train, y_val, y_test],
                                 [train_pred, val_pred, test_pred])
    results_df.insert(0, 'model', 'med')
    return results_df, model



############## MULTIMODAL MODEL ####################

def copy_model(model: keras.Model) -> keras.Model:
    model_copy = keras.models.clone_model(model)
    model_copy.build(model.input_shape)
    model_copy.set_weights(model.get_weights())
    return model_copy


def build_multimodal_model(model_dict: Dict[str, keras.Model]) -> keras.Model:
    um_inputs, um_outputs = [], []
    for name, model in model_dict.items():
        model = copy_model(model)
        
        feature_layer = model.get_layer('feature_layer')
        model = keras.Model(model.input, feature_layer.output)
        model._name = f'{name}_model'
        um_input = keras.layers.Input(shape=model.input_shape[1:], name=name)
        um_inputs.append(um_input)
        um_outputs.append(model(um_input))
    
    x = keras.layers.concatenate(um_outputs, name='concat_layer')
    x = keras.layers.Dropout(0.2, name='dropout_0')(x)
    x = keras.layers.Dense(8, kernel_regularizer=keras.regularizers.l2(0.01), kernel_initializer='he_normal', name='dense_0')(x)
    x = keras.layers.LeakyReLU()(x)
    
    mort_pred = keras.layers.Dense(1, activation='sigmoid', name='mortality')(x)

    mm_model = keras.Model(inputs=um_inputs, outputs=mort_pred)
    try: 
        mm_model.get_layer('hip_model').trainable = False
        mm_model.get_layer('chest_model').trainable = False
    except ValueError:
        pass
    return mm_model


def train_multimodal_model(um_model_dict: Dict[str, keras.Model],
                           target_dict: Dict[str, pd.DataFrame],
                           verbose=0) -> Tuple[pd.DataFrame, keras.Model]:
    um_models = list(um_model_dict.keys())
    
    generators = {
        'train': custom_generators.MultimodalGenerator(target_dict['train'].copy(), um_models) ,
        'val': custom_generators.MultimodalGenerator(target_dict['val'].copy(), um_models, shuffle=False),
        'test': custom_generators.MultimodalGenerator(target_dict['test'].copy(), um_models, shuffle=False)
    }
    
    model = build_multimodal_model(um_model_dict)
    extracted_feature_dict = extract_concat_features(model, generators)
    partial_model = train_final_layers(model, target_dict, extracted_feature_dict, lr=5e-2)
    
    # Finetuning
    model.compile(loss=custom_binary_entropy_loss(target_dict['train']['overl30d'].copy()), 
                optimizer=optimizers.Adam(5e-3), 
                metrics=get_unimodal_metrics())
    
    model.fit(
        generators['train'],
        epochs=5,
        validation_data=generators['val'],
        verbose=verbose
    )
    
    
    generators['train'].shuffle = False
    train_pred = model.predict(generators['train'], verbose=0)
    val_pred = model.predict(generators['val'], verbose=0)
    test_pred = model.predict(generators['test'], verbose=0)
    results_df = compute_metrics([generators['train'].get_labels(), generators['val'].get_labels(), generators['test'].get_labels()],
                                 [train_pred, val_pred, test_pred])
    return results_df, model




def train_final_layers(model: keras.Model,
                       target_dict: Dict[str, pd.DataFrame],
                       data_dict: Dict[str, pd.DataFrame],
                       lr: float= 1e-3) -> keras.Model:
    y_train, x_train = target_dict['train'].align(data_dict['train'], axis=0)
    y_val, x_val = target_dict['val'].align(data_dict['val'], axis=0)
    y_test, x_test = target_dict['test'].align(data_dict['test'], axis=0)

    # replace overl30d with target column name
    y_train = y_train['overl30d']
    y_val = y_val['overl30d']
    y_test = y_test['overl30d']
    
    
    prediction_model = build_partial_model(model)
    prediction_model.compile(loss=custom_binary_entropy_loss(y_train.copy()), 
            optimizer=optimizers.Adam(lr), 
            metrics=get_unimodal_metrics())
    prediction_model.fit(x=x_train.to_numpy(),
                         y=y_train.to_numpy(),
                         batch_size=32,
                         epochs=100,
                         validation_data=(x_val.to_numpy(), y_val.to_numpy()),
                         callbacks=get_callbacks(mmodal=False),
                         verbose=0,
                         shuffle=True)
    
    return prediction_model

