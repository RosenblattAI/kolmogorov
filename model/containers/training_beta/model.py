import os
import boto3
import wandb
import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback
from DatasetGenerator import DatasetGenerator

DEFAULT_DATASET_DIRECTORY = 'dataset'

def train():
    """Use the wandb configuration to create a model and train it"""
    # get the configuration settings from wandb
    config = wandb.config 
    
    tr_gen = DatasetGenerator(myConfig, shuffle=False, multitask=False)
    
    # get the datasets and relevant variables
    train_ds, train_steps, val_ds, val_steps, classes = _load_dataset(config)
    
    # TODO: be smarter about this
    img_shape = (224,224,3)
    
    # Setup a multi-gpu distributed batch training strategy
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        # handle if model is not found
        try:
            model_cl = getattr(tf.keras.applications, config['model'])
        except AttributeError:
            raise SyntaxError(f'Model "{config["model"]}" was not found, please adjust your sweep configuration.')

        # get pretrained model with top removed
        model = model_cl(
            include_top=False, 
            weights='imagenet' if config['imagenet'] else None, 
            input_shape=img_shape
        )

        # add data augmentation
        model = add_preprocessing(model)

        # build model to config specifications
        model = build_model(
            model, 
            config, 
            len(classes)
        )

        # may replace with wandb equivalent
        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            verbose=1
        )

        # setup checkpointing between epochs
        final_model_path = os.path.join(wandb.run.dir,'/checkpoints/final.ckpt')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=final_model_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )


        # compile the model
        model.compile(
            optimizer=opt,
            metrics=mets,
            loss=loss_fn, 
            loss_weights=loss_ws
        )

        # wandb sample logging
        tr_batch = next(train_ds)
        if config['orientation']:
            tr_image_batch, (tr_label_batch, tr_orientation_batch) = tr_batch
            log_batch(tr_image_batch.numpy(), tr_label_batch.numpy(), classes, orientation_batch=tr_orientation_batch.numpy())
        else:
            tr_image_batch, tr_label_batch = tr_batch
            log_batch(tr_image_batch.numpy(), tr_label_batch.numpy(), classes)

        # perfom training
        model.fit( 
            train_ds,
            steps_per_epoch=train_steps, 
            validation_data=val_ds,
            validation_steps=val_steps,
            epochs=config['epochs'], 
            callbacks=[earlystop, 
                       WandbCallback(),
                       model_checkpoint_callback
                      ]
        )

        # save the best model checkpoint to wandb
        wandb.save(final_model_path)


def log_batch(image_batch, label_batch, classes, orientation_batch=None):
    """Logs 16 labeled samples to Wandb"""
    if orientation_batch is None:
        examples = [
            wandb.Image(
                image_batch[n], 
                caption=f'class: {tf.boolean_mask(classes, label_batch[n])[0].numpy().decode()}'
            ) for n in range(16)
        ]
    else:
        examples = [
            wandb.Image(
                image_batch[n], 
                caption=f'class: {tf.boolean_mask(classes, label_batch[n])[0].numpy().decode()}\norientation: {(orientation_batch[n] * 360.0) + 180.0}'
            ) for n in range(16)
        ]
    wandb.log({'examples': examples})


def add_preprocessing(base_model):
    """ Adds a RandomTranslation layer to the front of the model"""
    model  = tf.keras.models.Sequential([
        base_model.input,
        tf.keras.layers.experimental.preprocessing.RandomTranslation(
            height_factor=0.4, 
            width_factor=0.4, 
            fill_mode='constant'
        ),
        tf.keras.layers.experimental.preprocessing.Normalization(),
        base_model
    ])
    return model


 def _corruption_test(self):
        print('DatasetGenerator: testing for corrupt classes...')
        passed = True
        for c in self.classes:
            rem_id = lambda x: x[x.index('_')+1:]
            tr_fnames = set(map(rem_id, next(os.walk(f'{path}/training/{c}'))[2]))
            val_fnames = set(map(rem_id, next(os.walk(f'{path}/validation/{c}'))[2]))
            if (tr_fnames == tr_fnames - val_fnames):
                print(f'DatasetGenerator: {c} passed')
            else:
                passed = False
                print(f'DatasetGenerator: {c} failed')
                print(f'DatasetGenerator: num_train: {len(tr_fnames)}')
                print(f'DatasetGenerator: num_val: {len(val_fnames)}')
                print(f'DatasetGenerator: num_intersect: {val_fnames.intersection(tr_fnames)}')
        return passed
    

def build_model(base_model, config, num_classes):
    """"Takes model body and intializes head with Wandb.config hyperparameters"""
    output_lys = []
    
    x = base_model.output
    
    try:
        base_out_ly = getattr(tf.keras.layers, config['base_output_setting'])
    except AttributeError:
        raise SyntaxError(f'Layer "{config["base_output_setting"]}" was not found, please adjust your sweep configuration.')
            
    # TODO: add check for activation_fn
    try:
        activation_fn = getattr(tf.keras.activations, config['activation'])
    except AttributeError:
        raise SyntaxError(f'Activation function "{config["base_output_setting"]}" was not found, please adjust your sweep configuration.')
    
    x = base_out_ly()(x)
    c = tf.keras.layers.Dense(config['hidden_classification_ly'], activation=config['activation'])(x) 
    c = tf.keras.layers.Dropout(config['dropout_classification_ly'])(c)
    classification = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification')(c)
    output_lys.append(classification)
    
    if config['orientation']:
        o = tf.keras.layers.Dense(config['hidden_orientation_ly'], activation=config['activation'])(x) 
        o = tf.keras.layers.Dropout(config['dropout_orientation_ly'])(o)
        orientation = tf.keras.layers.Dense(3, activation='sigmoid', name='orientation')(o)
        output_lys.append(orientation)
        
    return tf.keras.Model(inputs=base_model.input, outputs=output_lys)


if __name__=='__main__':
    gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    
    # TODO(developer): load the config from the best model from s3
    config_defaults = {
        'optimizer' : 'Adam',
        'batch_size' : 32 if gpus else 16,
        'epochs' : 1,
        'activation' : 'relu',
        'hidden_classification_ly' : 1024, 
        'hidden_orientation_ly' : 1024,
        'dropout_classification_ly' : 0.5,
        'dropout_orientation_ly' : 0.5,
        'orientation' : 1,
        'classification_weight': 1.0,
        'orientation_weight': 1.0,
        'model': 'EfficientNetB4',
        'base_output_setting': 'GlobalAveragePooling2D',
        'imagenet': 1
    }
    
    wandb.init(config=config_defaults)
    train()
    