import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics, optimizers, callbacks

import pandas as pd
import numpy as np

import sklearn.metrics as skmetrics
from sklearn.utils.class_weight import compute_class_weight

from typing import List, Dict


def get_callbacks(mmodal=False) -> List[callbacks.Callback]:
    if mmodal:
        early_stop = callbacks.EarlyStopping(monitor='val_mortality_auc', mode = 'max', patience=10, verbose=0, min_delta=1e-4, restore_best_weights = True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_mortality_auc', mode = 'max', factor=0.5, patience=5, verbose=0, min_delta=1e-4)
    else:
        early_stop = callbacks.EarlyStopping(monitor='val_auc', mode = 'max', patience=10, verbose=0, min_delta=1e-4, restore_best_weights = True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_auc', mode = 'max', factor=0.5, patience=5, verbose=0, min_delta=1e-4)
    return [early_stop, reduce_lr]


def get_unimodal_metrics() -> List[metrics.Metric]:
    return [metrics.Precision(name='precision'), 
            metrics.Recall(name='recall'),
            metrics.AUC(name='auc')]
    
def get_comp_metrics() -> List[metrics.Metric]:
    return [
        metrics.AUC(name='auc', multi_label=True),
        metrics.AUC(curve='pr', name='pr', multi_label=True)
    ]
    
def compute_metrics(y_true_list: list, y_pred_list: list,  labels=['train', 'val', 'test']) -> pd.DataFrame:
    results = []
    for y_true, y_pred, split in zip(y_true_list, y_pred_list, labels):
        fpr, tpr, thresholds = skmetrics.roc_curve(y_true, y_pred)
        J = tpr - fpr
        best_thresh = thresholds[np.argmax(J)]
        
        y_pred_bin = [1 if x > best_thresh else 0 for x in y_pred]
        precision, recall, thresholds = skmetrics.precision_recall_curve(y_true, y_pred)
        
        results.append({
            'split': split,
            'auc': skmetrics.roc_auc_score(y_true, y_pred),
            'pr': skmetrics.auc(recall, precision),
            'precision': skmetrics.precision_score(y_true, y_pred_bin),
            'recall': skmetrics.recall_score(y_true, y_pred_bin),
            'f1': skmetrics.f1_score(y_true, y_pred_bin)
        })
    
    return pd.DataFrame.from_records(results)

@tf.autograph.experimental.do_not_convert
def custom_binary_entropy_loss(y_true: pd.Series):
    class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=y_true.to_numpy())
    cweight_dict = {
        0: class_weights[0],
        1: class_weights[1]
    }
    @tf.autograph.experimental.do_not_convert
    def inner_loss(y_true, y_pred):
        y_true = keras.backend.cast(y_true, y_pred.dtype)
        weights = (cweight_dict[0]**(1-y_true)) * (cweight_dict[1]**(y_true))
        loss = weights * keras.backend.binary_crossentropy(y_true, y_pred)
        return loss   
    return inner_loss

def get_class_weights(y_train: pd.Series):
    class_weights = compute_class_weight(class_weight='balanced', classes=y_train.unique(), y=y_train.to_numpy())
    return {
        0: class_weights[0],
        1: class_weights[1]
    }

@tf.autograph.experimental.do_not_convert
def weighted_multilabel_loss(y_train: pd.DataFrame):
    temp_y_train = y_train.drop(columns='overl30d')
    number_dim = np.shape(temp_y_train.to_numpy())[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
            weights[i] = compute_class_weight(class_weight='balanced', classes=[0,1], y=temp_y_train.to_numpy()[:, i])

    @tf.autograph.experimental.do_not_convert     
    def weighted_loss(y_true, y_pred):
        if weights is None: return 0
        y_true = keras.backend.cast(y_true, y_pred.dtype)
        return keras.backend.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*keras.backend.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss



def build_partial_model(model: keras.Model) -> keras.Model:
    layer_names = [l.name for l in model.layers]
    final_layers = layer_names[layer_names.index('concat_layer')+1:]

    inputs = keras.layers.Input(shape=model.get_layer('concat_layer').output_shape[1:], name='concat_out')
    x = inputs
    for l in final_layers:
        x = model.get_layer(l)(x)

    return keras.Model(inputs=inputs, outputs=x)


def extract_concat_features(model: keras.Model, 
                            generators: Dict[str, keras.utils.Sequence]) -> Dict[str, pd.DataFrame]:
    concat_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer('concat_layer').output
    )
    feature_dict = {}
    for split, gen in generators.items():
        temp_index = gen.get_ids()
        feature_dict[split] = pd.DataFrame(data=concat_model.predict(gen), index=temp_index)
    
    return feature_dict