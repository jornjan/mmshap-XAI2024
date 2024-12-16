import tensorflow as keras

import numpy as np
import pandas as pd

from shap import KernelExplainer, kmeans

import custom_generators
from util_functions import *

from typing import Tuple, Dict

def extract_mmodal_shapley(partial_model: keras.Model, 
                           data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, float]:
    x_train = data_dict['train']
    x_test = data_dict['test']
    
    explainer = KernelExplainer(partial_model.predict, kmeans(x_train, 100), link='identity')
    sv_test = np.squeeze(np.array(explainer.shap_values(x_test.to_numpy())))
    sv_test = np.squeeze(np.array(sv_test))
    sv_test = pd.DataFrame(sv_test, index=x_test.index)
    
    return sv_test, explainer.fnull


def rename_sv_cols(model: keras.Model, sv_values: pd.DataFrame) -> pd.DataFrame:
    col_rename_dict = {}
    base_col = 0
    for temp_layer in model.get_layer('concat_layer').inbound_nodes[0].inbound_layers:    
        name = temp_layer.name.split('_')[0]
        n_feat = temp_layer.output_shape[1]
        col_rename_dict.update({base_col+i:f'{name}-{i}' for i in range(n_feat)})
        
        base_col+=n_feat
        
    return sv_values.rename(columns=col_rename_dict)
    
    
def compute_modality_importance(input: pd.DataFrame):
    mean_abs_importance = input.abs().mean(axis=0).T.reset_index()
    mean_abs_importance['modality'] = mean_abs_importance['index'].apply(lambda x: x.split('-')[0])
    mean_abs_importance = mean_abs_importance.drop(columns='index')
    modality_importance = mean_abs_importance.groupby('modality').sum()
    
    modality_importance = modality_importance.rename(columns={0: 'sum'})
    modality_importance['perc'] = (100*modality_importance['sum'] / modality_importance['sum'].sum()).round(1)

    return modality_importance


def extract_feature_importance(model: keras.Model,
                               target_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, float]:
    
    um_models = [l.name for l in model.input]
    generators = {
        'train': custom_generators.MultimodalGenerator(target_dict['train'].copy(), um_models) ,
        'val': custom_generators.MultimodalGenerator(target_dict['val'].copy(), um_models, shuffle=False),
        'test': custom_generators.MultimodalGenerator(target_dict['test'].copy(), um_models, shuffle=False)
    }

    extracted_feature_dict = extract_concat_features(model, generators)
    partial_model = build_partial_model(model)
    sv_test, fnull = extract_mmodal_shapley(partial_model, extracted_feature_dict)
    sv_test = rename_sv_cols(model, sv_test)
    return sv_test, fnull


def extract_static_shap(model: keras.Model,
                        target_dict: Dict[str, pd.DataFrame],
                        sv_pred: pd.DataFrame):
    fname = 'PATH/TO/IMPUTED/STATIC/DATA.pkl'
    raise NotImplementedError('Please add the local path to imputed static data')

    temp_static_data = pd.read_pickle('fname')
    temp_static_model = model.get_layer('static_model')
    
    y_train = target_dict['train'].copy()
    y_test = target_dict['test'].copy()
    
    X_train = temp_static_data.loc[y_train.index.tolist()]
    X_test = temp_static_data.loc[y_test.index.tolist()]
    
    explainer = KernelExplainer(temp_static_model.predict, kmeans(X_train, 100), link='identity')
    temp_static_shap = np.squeeze(np.array(explainer.shap_values(X_test.to_numpy())))
    temp_static_shap = np.squeeze(np.array(temp_static_shap))
    temp_static_shap = np.moveaxis(temp_static_shap, 1, 0)
    
    extracted_features = temp_static_model.predict(X_test.to_numpy())
    
    sv_pred = sv_pred.copy()
    sv_pred = sv_pred.loc[X_test.index]
    static_feature_shap = sv_pred.filter(like='static').to_numpy()

    temp_answer = temp_static_shap / (extracted_features[:,:, np.newaxis] + 1e-7)       # Percentage of feature value
    temp_answer = temp_answer * static_feature_shap[:, :, np.newaxis]                   # Weighted contribution to prediction
    temp_answer = temp_answer.sum(axis=1)                                               # Sum accross all intermediate features
    X_test_shap = pd.DataFrame(temp_answer, columns=X_test.columns, index=X_test.index) # Convert to DataFrame
    
    fnull_test_shap = explainer.fnull / extracted_features
    fnull_test_shap = fnull_test_shap * static_feature_shap
    fnull_test_shap = fnull_test_shap.sum(axis=1)
    
    X_test_shap['fnull'] = fnull_test_shap
    
    return X_test_shap
