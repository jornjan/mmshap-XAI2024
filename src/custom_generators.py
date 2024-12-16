import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences

from typing import List

class ImageDataGenerator(Sequence):
    def __init__(self,
                 label_df: pd.Series,
                 modality: str,
                 batch_size=32,
                 shuffle=True) -> None:
        
        self.label_df = label_df.copy().sort_index()
        self.cid_list = label_df.index.tolist()
        
        self.modality = modality        
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()
        
    def __len__(self):
        "Number of batches per epoch"
        return int(np.ceil(len(self.cid_list) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        if self.shuffle:
            np.random.shuffle(self.cid_list)
            
    def get_labels(self):
        return self.label_df.loc[self.cid_list].to_numpy()
    
    def __getitem__(self, index):
        cid_batch = self.cid_list[index*self.batch_size:(index+1)*self.batch_size]
        images_batch = []
        for cid in cid_batch:
            fname = 'PATH/TO/PATIENT/IMAGE.npy'
            raise NotImplementedError('Please add the local path to Image data')
            temp_img = image.load_img(fname,
                                      target_size=(224,224),
                                      interpolation='bicubic')

            input_arr = image.img_to_array(temp_img)
            images_batch.append(input_arr)
        
        X = np.array(images_batch)
        y = self.label_df.loc[cid_batch].to_numpy()
        return X, y



class VitalsGenerator(Sequence):
    def __init__(self,
                 label_df: pd.Series,
                 batch_size=32,
                 shuffle=True) -> None:
        self.label_df = label_df.copy().sort_index()
        self.cid_list = label_df.index.tolist()

          
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()
    
    def __len__(self):
        "Number of batches per epoch"
        return int(np.ceil(len(self.cid_list) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        if self.shuffle:
            np.random.shuffle(self.cid_list)
            
    def get_labels(self):
        return self.label_df.loc[self.cid_list].to_numpy()
    
    
    def __getitem__(self, index):
        cid_batch = self.cid_list[index*self.batch_size:(index+1)*self.batch_size]
        X = []
        for cid in cid_batch:
            fname = 'PATH/TO/PATIENT/VITALS.npy'
            raise NotImplementedError('Please add the local path to vitals data')
            X.append(np.load(fname))
            
        X = pad_sequences(X, padding='pre', dtype=float, value=-10.)
        y = self.label_df.loc[cid_batch].to_numpy()
        return X, y
    


class MultimodalGenerator(Sequence):
    def __init__(self,
                 label_df: pd.DataFrame,
                 models_list: List[str],
                 batch_size=32,
                 shuffle=True) -> None:
        self.mort_df  = label_df['overl30d'].copy().sort_index()
        self.comp_df = label_df.copy().drop(columns='overl30d').sort_index()
        self.cid_list = label_df.index.tolist()
        
        fname = 'PATH/TO/PATIENT/DATA.pkl'
        raise NotImplementedError('Please add the local path to static data')
        self.static_df = pd.read_pickle(fname)

        fname = 'PATH/TO/PATIENT/DATA.pkl'
        raise NotImplementedError('Please add the local path to medication data')
        
        self.med_df =  pd.read_pickle(fname)
        
        self.models_list = models_list
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()
    
    def __len__(self):
        "Number of batches per epoch"
        return int(np.ceil(len(self.cid_list) / self.batch_size))
    

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        if self.shuffle:
            np.random.shuffle(self.cid_list)
            
    def get_labels(self):
        return self.mort_df.loc[self.cid_list].to_numpy()
    
    def get_ids(self):
        return self.mort_df.loc[self.cid_list].index.tolist()
        
        
    def __getitem__(self, index):
        cid_batch = self.cid_list[index*self.batch_size:(index+1)*self.batch_size]
        X = {}
        if 'static' in self.models_list:
            X['static'] = self.static_df.loc[cid_batch].to_numpy()
        if 'hip' in self.models_list:
            X['hip'] = self.__getimages__(cid_batch, 'hip')
        if 'chest' in self.models_list:
            X['chest'] = self.__getimages__(cid_batch, 'chest')
        if 'vitals' in self.models_list:
            X['vitals'] = self.__getvitals__(cid_batch)
        if 'med' in self.models_list:
            X['med'] = self.med_df.loc[cid_batch].to_numpy()
                
        y = self.__get_targets__(cid_batch)
        return X, y
    
    
    def __getimages__(self, cid_batch, modality: str):    
        images_batch = []
        for cid in cid_batch:
            fname = 'PATH/TO/PATIENT/IMAGE.npy'
            raise NotImplementedError('Please add the local path to Image data')
            temp_img = image.load_img(fname,
                                      target_size=(224,224),
                                      interpolation='bicubic')

            input_arr = image.img_to_array(temp_img)
            images_batch.append(input_arr)
        X = np.array(images_batch)
        return X
    
    def __getvitals__(self, cid_batch):
        X = []
        for cid in cid_batch:
            fname = 'PATH/TO/PATIENT/VITALS.npy'
            raise NotImplementedError('Please add the local path to vitals data')
            X.append(np.load(fname))
            
        X = pad_sequences(X, padding='pre', dtype=float, value=-10.)
        return X
    
    def __get_targets__(self, cid_batch):
        return self.mort_df.loc[cid_batch].to_numpy()