import os
import numpy as np
import tensorflow as tf
from Model import Model
from DatasetGenerator import DatasetGenerator

class TrainingPipeline:
    
    
    def __init__(self, conf):
        self.conf = conf
        self.tr_path = 'dataset/training'
        self.val_path = 'dataset/validation'
        
        
    def dataset_contamination_test(self):
        print('Pipeline: testing for corrupt classes...')
        passed = True
        for c in self.classes:
            rem_id = lambda x: x[x.index('_')+1:]
            tr_fnames = set(map(rem_id, next(os.walk(f'{self.tr_path}/{c}'))[2]))
            val_fnames = set(map(rem_id, next(os.walk(f'{self.val_path}/{c}'))[2]))
            if (tr_fnames == tr_fnames - val_fnames):
                print(f'Pipeline: {c} passed')
            else:
                passed = False
                print(f'Pipeline: {c} failed')
                print(f'Pipeline: num_train: {len(tr_fnames)}')
                print(f'Pipeline: num_val: {len(val_fnames)}')
                print(f'Pipeline: num_intersect: {val_fnames.intersection(tr_fnames)}')
        return passed
    
    
    def get_ds_conf(self):
        batch = self.config['batch_size']
        num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
        if num_gpus > 1:
            batch *= num_gpus
            print(f'Pipeline: using batch sizes of {batch} equally distributed on {num_gpus} GPUs...')
        else:
            print(f'Pipeline: using batch sizes of {batch} on a single GPU...')
        multitask = self.config['orientation']
        
        ds_conf = {
            'batch': batch,
            'multitask': multitask
        }
        
        return ds_conf
    
    
    def get_model_compile_conf(self):
        # handle if optimizer is not found
        try:
            opt = getattr(tf.keras.optimizers, config['optimizer'])()
        except AttributeError:
            raise SyntaxError(f'Optimizer "{config["optimizer"]}" was not found, please adjust your sweep configuration.')

        # loss funciton(s)             
        if config['orientation']:
            loss_fn= {
                'classification': 'categorical_crossentropy',
                'orientation': 'mse'
            }
        else:
            loss_fn = {'classification': 'categorical_crossentropy'}

        # loss weight(s)
        if config['orientation']:
            loss_ws = {
                'classification': config['classification_weight'], 
                'orientation': config['orientation_weight']
            } 
        else:
            loss_ws = None

        # metric name
        if config['orientation']:
            mets = {'classification': 'accuracy'}
        else:
            mets = [
                tf.keras.metrics.CategoricalAccuracy(name='classification_accuracy'),
                tf.keras.metrics.CategoricalCrossentropy(name='classification_loss')
            ]
            
        compile_config = {
            'optimizer': opt, 
            'loss': loss_fn, 
            'loss_weights': loss_ws,
            'metrics': mets
        }
    
        return compile_conf
    
    
    def execute(self):
        # setup dataset
        ds_conf = get_ds_conf()
        tr_ds_gen = DatasetGenerator('dataset/training', **ds_conf)
        val_ds_gen = DatasetGenerator('dataset/validation', **ds_conf)
        
        # run tests
        dataset_contamination_test()
        
        # setup model
        m_conf = get_model_conf()
        model = Model(**m_conf)
        
        # compile the model
        c_conf = get_compile_conf()
        model.compile(**c_conf)
        
class InferencePipeline:
    pass