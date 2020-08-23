import os
import wandb
import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback

DEFAULT_DATASET_DIRECTORY = 'dataset'

def train():
    """Use the wandb configuration to create a model and train it"""
    # get the configuration settings from wandb
    config = wandb.config 
    
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

        # validation loss early stop callback
        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        # perfom training
        model.fit( 
            train_ds,
            steps_per_epoch=train_steps, 
            validation_data=val_ds,
            validation_steps=val_steps,
            epochs=config['epochs'], 
            callbacks=[
                earlystop, 
                checkpoint_callback,
                wandb_logging,
            ]
        )
        
        # TODO: remove translation layer
        model.load_weights(checkpoint_filepath)
        print('Calculating Confusion Matrix and AUC-ROC Curve using best weigths')
        calculate_metrics(val_ds, val_steps, classes, model)
        
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
    
    
def calculate_metrics(gen, steps, classes, model):
    """Logs a confussion matrix and ROC curves for 'step' number of batches from a dataset generator"""
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
        orientation = tf.keras.layers.Dense(6, activation='sigmoid', name='orientation')(o)
        output_lys.append(orientation)
        
    return tf.keras.Model(inputs=base_model.input, outputs=output_lys)


def _extract_orientation(filename):
    o_string = filename.split('_')[2]
    return [float(val) for val in o_string.split(',')]


def dataset_generator(folder, classes, batch=16, shuffle=True, multitask=False):
    """Generates dataset batches"""
    file_gen = os.walk(folder)
    _ = next(file_gen)[1]
    one_hot_encoder = dict(zip(classes, np.eye(len(classes))))
    if multitask:
        dataset = [[f'{class_[0]}/{filename}', _extract_orientation(filename), one_hot_encoder[class_[0].split('/')[-1]]] for class_ in file_gen for filename in class_[2]]
    else:
        dataset = [[f'{class_[0]}/{filename}', one_hot_encoder[class_[0].split('/')[-1]]] for class_ in file_gen for filename in class_[2]]

    i = 0

    while True:   
        if shuffle and i == 0:
            np.random.shuffle(dataset)

        if multitask:
            orients = []

        images, one_hots = [], []

        try:
            for sample in range(batch):
                if multitask:
                    path, orient, one_hot = dataset[i*batch + sample]
                else:
                    path, one_hot = dataset[i*batch + sample]
                    
                with open(path, 'rb') as img:
                    f = img.read()
                    b_img = bytes(f)
                
                img = tf.io.decode_image(b_img)
                img = tf.image.convert_image_dtype(img, tf.float32)
                images.append(img)
                
                if multitask:
                    orient = tf.convert_to_tensor(_extract_orientation(path))
                    orient = np.deg2rad(orient)
                    orient = tf.concat([tf.math.sin(orient), tf.math.cos(orient)], 0)
                    orients.append(orient)

                one_hot = tf.convert_to_tensor(one_hot)
                one_hots.append(one_hot)

                i += 1
        except IndexError:
            i = 0
            continue

        if multitask:
            yield tf.convert_to_tensor(images), [tf.convert_to_tensor(one_hots), tf.convert_to_tensor(orients)]
        else:
            yield tf.convert_to_tensor(images), tf.convert_to_tensor(one_hots)


def _corruption_test(classes):
    path = DEFAULT_DATASET_DIRECTORY
    print('model.py: testing for corrupt classes...')
    passed = True
    for c in classes:
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


def _load_dataset(config):
    path = DEFAULT_DATASET_DIRECTORY
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    batch = config['batch_size'] * num_gpus if num_gpus else config['batch_size']
    if num_gpus > 1:
        print(f'model.py: using batch sizes of {batch} equally distributed on {num_gpus} GPUs...')
    else:
        print(f'model.py: using batch sizes of {batch} on a single GPU...')

    classes = sorted(next(os.walk(f'{path}/training'))[1])

    if not _corruption_test(classes):
        # TODO: Replace with custom error
        raise RuntimeError('Validation dataset contamination detected. Please update the dataset.')
    try:
        num_train = sum([len(next(os.walk(f'{path}/training/{c}'))[2]) for c in classes])
        num_val = sum([len(next(os.walk(f'{path}/validation/{c}'))[2]) for c in classes])
    except Exception as e:
        print(f'model.py: There was an exception getting the number of training and validation samples (are they all downloaded?): {str(e)}')

    train_steps = int(np.ceil(num_train/batch))
    val_steps = int(np.ceil(num_val/batch))
    train_ds = dataset_generator(f'{path}/training', classes, batch=batch, multitask=config['orientation'])
    val_ds = dataset_generator(f'{path}/validation', classes, batch=batch, multitask=config['orientation'])
    return train_ds, train_steps, val_ds, val_steps, classes


if __name__=='__main__':
    gpus = len(tf.config.experimental.list_physical_devices('GPU'))

    # TODO(developer): load the config from the best model from s3
    config_defaults = {
        'optimizer' : 'Adam',
        'batch_size' : 64 if gpus else 16,
        'epochs' : 10,
        'activation' : 'relu',
        'hidden_classification_ly' : 256,
        'hidden_orientation_ly' : 256,
        'dropout_classification_ly' : 0.3,
        'dropout_orientation_ly' : 0.5,
        'orientation' : 0,
        'classification_weight': 1.0,
        'orientation_weight': 1.0,
        'model': 'ResNet50V2',
        'base_output_setting': 'GlobalAveragePooling2D',
        'imagenet': 1
    }

    wandb.init(config=config_defaults)
    train()
