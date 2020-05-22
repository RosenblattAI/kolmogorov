import os
import json
import wandb
import pathlib
import argparse
import callbacks
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def model():
    """Generate a simple model using the Keras API for Tensorflow"""
    args, unknown = _parse_args()
    
    config = wandb.config # When Sweeping, wandb.config will be updated every session
    
    train_ds, train_steps, val_ds, val_steps, labels = _load_dataset(args.train, config['batch_size'])
    
    if config['model'] == 'ResNet':
        model = tf.keras.applications.resnet.ResNet50(include_top=False, 
                                                      weights='imagenet', 
                                                      input_shape=(224,224,3), 
                                                      classes=len(labels))
    elif config['model'] == 'VGG':
        model = tf.keras.applications.vgg16.VGG16(include_top=False, 
                                                      weights='imagenet', 
                                                      input_shape=(224,224,3), 
                                                      classes=len(labels))
    
    model = build_finetune_model(model, 
                                 [config['dropout'], config['dropout']], 
                                 [config['hidden'], config['hidden']],
                                  config['activation'], len(labels))
    
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1)
    
    if config['optimizer'] == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    elif config['optimizer'] == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
    elif config['optimizer'] == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])
    
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, epochs=config['epochs'], callbacks=[earlystop, 
              wandb.keras.WandbCallback()], steps_per_epoch=train_steps, #callbacks=[callbacks.WandbBatchHistory()]
              validation_data=val_ds, validation_steps=val_steps, verbose=0)
    
    # maybe do some predictions
    
    return model


def build_finetune_model(base_model, dropouts, fc_layers, activation, num_classes):

    x = base_model.output
    x = tf.keras.layers.Flatten()(x)

    for fc, drop in zip(fc_layers, dropouts):
        x = tf.keras.layers.Dense(fc, activation=activation)(x) 
        x = tf.keras.layers.Dropout(drop)(x)

    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=base_model.input, outputs=predictions)


def get_label(file_path, labels):
    parts  = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == labels


def decode_img(img):
    
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    #####  IMAGE BLURRING GOES HERE #####
    
    return img


def process_file(file_path):
    
    label = get_label(file_path)
    img   = tf.io.read_file(file_path)
    img   = decode_img(img)
    
    return img, label


def prepare_for_training(ds, size, cache=True, shuffle_buffer_size=1000):
    """
    This is a small dataset, only load it once, and keep it in memory.
    use `.cache(filename)` to cache preprocessing work for datasets that don't
    fit in memory.
    """
    
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
    else:
        ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def _load_dataset(base_dir):
    """Load Satellite training data"""
    
    # get the unique labels and number of images
    labels      = list(filter(lambda x: x != '.DS_Store', os.listdir(image_dir)))
    num_images  = len(list(pathlib.Path(base_dir).glob('*/*.png')))
    
    # 80/20 train/test split
    num_train   = int(num_images * 0.8)
    num_val     = int(num_images * 0.2)
    
    # calculate steps for each epoch
    train_steps = int(np.ceil(num_train/wand.config['batch_size']))
    val_steps   = int(np.ceil(num_val/wand.config['batch_size']))
    
    #  create the full dataset
    ds_files    = tf.data.Dataset.list_files(str(pathlib.Path(base_dir)/'*/*.png'))
    
    labeled_ds  = ds_files.map(process_file(size=wand.config['batch_size']), # could fail here maybe
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    full_ds     = prepare_for_training(labeled_ds)
    
    # split the dataset 80/20
    train_ds    = full_ds.take(num_train)
    val_ds      = full_ds.skip(num_train)
    
    return train_ds, train_steps, val_ds, val_steps, labels


def _parse_args():    
    """Parse the arguments passed from wandb_setup.sh"""
    
    parser = argparse.ArgumentParser()

    # Set a default configuration for the model's hyperparameters
    config_defaults = {
        "learning_rate" : 1e-4,
        "epochs" : 10,
        "dropout" : 0.5,
        "hidden" : 1024,
        "batch_size" : 16,
        "optimizer" : "adam",
        "activation" : "relu",
        "model" : "ResNet"
    }
    
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--wandb-entity', type=str, default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--wandb-project', type=str, default=os.environ.get('WANDB_PROJECT'))
    parser.add_argument('--configuration', type=str, default=config_default)
    parser.add_argument('--wandb-group', type=str, default=None)
    parser.add_argument('--wandb-job-type', type=str, default=None)
    parser.add_argument('--wandb-sweep-id', type=str, default=None)
    
    return parser.parse_known_args()


if __name__ == "__main__":
    
    """Setup Wandb according to the arguments passed from wandb_setup.sh"""
    
    args, unknown = _parse_args()
    
    # Initalize Wandb using the arguments scraped by wandb_setup.sh
    wandb.init(entity   = args.wandb_entity,
               project  = args.wandb_project, 
               group    = args.wandb_group, 
               job_type = args.wandb_job_type, 
               config   = args.configuration)
    
    model()
    
    #if args.wandb_sweep_id is None:
    #    model()
    #else:
    #    wandb.agent(args.wandb_sweep_id, model)