import os
import wandb
import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback

DEFAULT_DATASET_DIRECTORY = 'dataset'

def train():
    """Use the wandb configuration to create a model and train it"""
    
    # lower backpropagation resolution
    if wandb.config['mixed_precision']:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy) 
    
    # get the datasets and relevant variables
    train_ds, val_ds, classes = _load_dataset()
    
    # TODO: be smarter about this
    img_shape = (224,224,3)
    
    # Setup a multi-gpu distributed batch training strategy
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        # handle if model is not found
        try:
            model_cl = getattr(tf.keras.applications, wandb.config['model'])
        except AttributeError:
            raise SyntaxError(f'Model "{wandb.config["model"]}" was not found, please adjust your sweep configuration.')

        # get pretrained model with top removed
        model = model_cl(
            include_top=False, 
            weights='imagenet' if wandb.config['imagenet'] else None,
            input_shape=img_shape
        )

        # add data augmentation
        model = add_preprocessing(model)

        # build model to config specifications
        model = build_model(
            model, 
            len(classes)
        )

        # handle if optimizer is not found
        try:
            opt = getattr(tf.keras.optimizers, wandb.config['optimizer'])()
        except AttributeError:
            raise SyntaxError(f'Optimizer "{wandb.config["optimizer"]}" was not found, please adjust your sweep configuration.')

        # loss funciton(s)             
        if wandb.config['orientation']:
            loss_fn= {
                'classification': 'categorical_crossentropy',
                'orientation': 'mse'
            }
        else:
            loss_fn = {'classification': 'categorical_crossentropy'}

        # loss weight(s)
        if wandb.config['orientation']:
            loss_ws = {
                'classification': wandb.config['classification_weight'], 
                'orientation': wandb.config['orientation_weight']
            } 
        else:
            loss_ws = None

        # metric name
        if wandb.config['orientation']:
            mets = {'classification': 'accuracy'}
        else:
            mets = [
                tf.keras.metrics.CategoricalAccuracy(name='classification_accuracy'),
                tf.keras.metrics.CategoricalCrossentropy(name='classification_loss')
            ]

        # compile the model
        model.compile(
            optimizer=opt,
            metrics=mets,
            loss=loss_fn, 
            loss_weights=loss_ws
        )

        # wandb sample logging
        tr_batch = next(iter(train_ds))
        if wandb.config['orientation']:
            tr_image_batch, (tr_label_batch, tr_orientation_batch) = tr_batch
            #tr_orientation_batch = cylcial2degrees(tr_orientation_batch.numpy())
            tr_orientation_batch = tr_orientation_batch.numpy()
            log_batch(tr_image_batch.numpy(), tr_label_batch.numpy(), classes, orientation_batch=tr_orientation_batch)
        else:
            tr_image_batch, tr_label_batch = tr_batch
            log_batch(tr_image_batch.numpy(), tr_label_batch.numpy(), classes)

        # wandb metric and best val_classification_accuracy logging
        wandb_logging = WandbCallback(
            monitor='val_classification_accuracy',
            mode='max',
            save_weights_only=True,
        )
        
        """
        # setup model checkpointing
        checkpoint_filepath = os.path.join(wandb.run.dir, "checkpoint/best_model")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor='val_classification_accuracy',
            verbose=1, 
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )
        """

        # validation loss early stop callback
        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        # perfom training
        model.fit( 
            train_ds,
            validation_data=val_ds,
            epochs=wandb.config['epochs'], 
            callbacks=[
                earlystop, 
                wandb_logging,
            ]
        )
        
        
        """
        # TODO: remove translation layer
        model.load_weights(checkpoint_filepath)
        print('Calculating Confusion Matrix and AUC-ROC Curve using best weigths')
        calculate_metrics(val_ds, val_steps, classes, model)
        """
        
"""
def cylcial2degrees(data):
    #Converts cyclical sin and cos pairs to degrees (reverse unit circle)
    axs = []
    num_features = data.shape[1]//2
    for axis in range(num_features):
        ax_sin = data[:,axis]
        ax_cos = data[:,axis+num_features]
        ax = np.arctan(np.divide(ax_sin, ax_cos))
        axs.append(ax)
    return np.degrees(np.array(axs).T)
""" 
    
"""
def calculate_metrics(gen, steps, classes, model):
    #Logs a confussion matrix and ROC curves for 'step' number of batches from a dataset generator
    labels = []
    preds = []
    if wandb.config['orientation']:
        orients = []
    
    for _ in range(steps):
        batch = next(gen)
        if wandb.config['orientation']:
            image_batch, (label_batch, orientation_batch) = batch
            class_batch, orient_batch = model.predict(image_batch)
            preds.append(np.argmax(class_batch, axis=1))
            orients.append(orient_batch)
        else:
            image_batch, label_batch = batch
            pred_batch = model.predict(image_batch)
            preds.append(pred_batch)
        labels.append(np.argmax(label_batch.numpy(), axis=1))
        
    y_true = np.array(labels).flatten()
    y_prob = np.concatenate(preds)
    y_pred = np.argmax(y_prob, axis=1).flatten()
    wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels=classes)
    wandb.log({'roc': wandb.plots.ROC(y_true, y_prob)}, classes)
   
    if wandb.config['orientation']:
         # TODO: support orientation
        pass
"""

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
                caption=f'class: {tf.boolean_mask(classes, label_batch[n])[0].numpy().decode()}\norientation: {orientation_batch[n]}'
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


def build_model(base_model, num_classes):
    """"Takes model body and intializes head with Wandb.config hyperparameters"""
    output_lys = []

    x = base_model.output

    try:
        base_out_ly = getattr(tf.keras.layers, wandb.config['base_output_setting'])
    except AttributeError:
        raise SyntaxError(f'Layer "{wandb.config["base_output_setting"]}" was not found, please adjust your sweep configuration.')
            
    # TODO: add check for activation_fn
    try:
        activation_fn = getattr(tf.keras.activations, wandb.config['activation'])
    except AttributeError:
        raise SyntaxError(f'Activation function "{wandb.config["base_output_setting"]}" was not found, please adjust your sweep configuration.')

    x = base_out_ly()(x)
    c = tf.keras.layers.Dense(wandb.config['hidden_classification_ly'], activation=wandb.config['activation'])(x) 
    c = tf.keras.layers.Dropout(wandb.config['dropout_classification_ly'])(c)
    classification = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification')(c)
    output_lys.append(classification)
    
    if wandb.config['orientation']:
        o = tf.keras.layers.Dense(wandb.config['hidden_orientation_ly'], activation=wandb.config['activation'])(x) 
        o = tf.keras.layers.Dropout(wandb.config['dropout_orientation_ly'])(o)
        orientation = tf.keras.layers.Dense(6, activation='sigmoid', name='orientation')(o)
        output_lys.append(orientation)
        
    return tf.keras.Model(inputs=base_model.input, outputs=output_lys)


def path2label(path, labels):
    label = tf.strings.split(path, os.path.sep)[-2].numpy()
    labels = labels.numpy()
    return labels == label


def path2img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def path2orientation(path):
    orientation_str = tf.strings.split(path, '_')[-7]
    orientation = tf.strings.to_number(tf.strings.split(orientation_str, ','))
    orientation = np.deg2rad(orientation)
    orientation = tf.concat([tf.math.sin(orientation), tf.math.cos(orientation)], 0)
    return orientation


def get_dataset(directory):
    labels = sorted(os.listdir(directory))
    files_ds = tf.data.Dataset.list_files(os.path.join(directory,'*/*.png'))
    
    if wandb.config['orientation']:
        orientation_ds = files_ds.map(
            lambda path: tf.py_function(
                path2orientation, [path], [tf.float32]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    img_ds = files_ds.map(
        lambda path: tf.py_function(
            path2img, [path], [tf.float32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    labels_ds = files_ds.map(
        lambda path: tf.py_function(
            path2label, [path, labels], [tf.float32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    outputs_ds = tf.data.Dataset.zip((labels_ds, orientation_ds)) if wandb.config['orientation'] else labels_ds
    ds = tf.data.Dataset.zip((img_ds, outputs_ds))
    ds = ds.cache()
    ds = ds.shuffle(files_ds.cardinality().numpy(), reshuffle_each_iteration=True)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(get_batch_size())
    return ds


def _corruption_test():
    path = DEFAULT_DATASET_DIRECTORY
    print('model.py: testing for corrupt classes...')
    passed = True
    for c in sorted(os.listdir(directory)):
        rem_id = lambda x: x[x.index('_')+1:]
        tr_fnames = set(map(rem_id, next(os.walk(f'{path}/training/{c}'))[2]))
        val_fnames = set(map(rem_id, next(os.walk(f'{path}/validation/{c}'))[2]))
        if (tr_fnames == tr_fnames - val_fnames):
            print(f'model.py: {c} passed')
        else:
            passed = False
            print(f'model.py: {c} failed')
            print(f'model.py: num_train: {len(tr_fnames)}')
            print(f'model.py: num_val: {len(val_fnames)}')
            print(f'model.py: num_intersect: {val_fnames.intersection(tr_fnames)}')
    return passed


def get_batch_size():
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
     wandb.run.summary['num_gpus'] = num_gpus
    batch_size = wandb.config['batch_size'] * num_gpus if num_gpus else wandb.config['batch_size']
    print(f'model.py: batch size: {batch_size}')
    
    if num_gpus > 1:
        print(f'model.py: using batch sizes of {batch_size} equally distributed on {batch_size} GPUs. Updating wandb.summary...')
        wandb.run.summary['distributed_batch_size'] = batch_size
    else:
        print(f'model.py: using batch sizes of {batch_size} on a single GPU...')
    return batch_size


def _load_datasets():
    path = DEFAULT_DATASET_DIRECTORY
    
    if sorted(os.listdir(f'{path}/training')) != sorted(os.listdir(f'{path}/validation')):
        # TODO: Replace with custom error
        raise RuntimeError('Training and validation folder do not contain the same classes. Please update the dataset.')
    else:
        classes = sorted(os.listdir(f'{path}/validation'))

    if not _corruption_test(classes):
        # TODO: Replace with custom error
        raise RuntimeError('Validation folder contamination detected. Please update the dataset.')

    train_ds = get_dataset(f'{path}/training')
    val_ds = get_dataset(f'{path}/validation')
    return train_ds, val_ds, classes


if __name__=='__main__':

    # TODO(developer): load the config from the best model from s3
    # TODO: adjust batch size to the size of the GPU
    config_defaults = {
        'optimizer' : 'Adam',
        'batch_size' : 96 if len(tf.config.experimental.list_physical_devices('GPU')) else 16,
        'epochs' : 10,
        'activation' : 'relu',
        'hidden_classification_ly' : 1000,
        'hidden_orientation_ly' : 1000,
        'dropout_classification_ly' : 0.3,
        'dropout_orientation_ly' : 0.5,
        'orientation' : 1,
        'classification_weight': 1.0,
        'orientation_weight': 1.0,
        'model': 'ResNet50V2',
        'base_output_setting': 'GlobalAveragePooling2D',
        'imagenet': 1
        'mixed_precision': 0,
    }

    wandb.init(config=config_defaults)
    train()
