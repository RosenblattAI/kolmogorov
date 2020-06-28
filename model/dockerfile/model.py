import s3fs
import wandb
import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback


def train():
    
    config = wandb.config 
    
    train_ds, train_steps, val_ds, val_steps, classes = _load_dataset(batch=config['batch_size'])
    
    img_shape = (224,224,3)
    
    if config['model'] == 'ResNet-50':
        model = tf.keras.applications.resnet.ResNet50(include_top=False, 
                                                      weights='imagenet', 
                                                      input_shape=img_shape, 
                                                      classes=len(classes))
    if config['model'] == 'ResNet-152':
        model = tf.keras.applications.resnet.ResNet152(include_top=False, 
                                                       weights='imagenet',
                                                       input_shape=img_shape, 
                                                       classes=len(classes))
    elif config['model'] == 'VGG-16':
        model = tf.keras.applications.vgg16.VGG16(include_top=False, 
                                                      weights='imagenet', 
                                                      input_shape=img_shape, 
                                                      classes=len(classes))
    elif config['model'] == 'VGG-19':
        model = tf.keras.applications.vgg19.VGG19(include_top=False, 
                                                      weights='imagenet', 
                                                      input_shape=img_shape, 
                                                      classes=len(classes))

    model = build_finetune_model(model, 
                                 [config['dropout_ly1'], config['dropout_ly2']], 
                                 [config['hidden_ly1'], config['hidden_ly2']],
                                  config['activation'], len(classes))
    
    # may replace with wandb equivalent
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1)
    
    if config['optimizer'] == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])
    
    model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')
    
    # Workaround for Issue #24
    tr_image_batch, tr_label_batch, _ = next(train_ds)
    log_batch(tr_image_batch.numpy(), tr_label_batch.numpy(), classes)

    model.fit(train_ds, 
              steps_per_epoch=train_steps, 
              validation_data=val_ds,
              validation_steps=val_steps,
              epochs=config['epochs'], 
              callbacks=[earlystop, WandbCallback()]
             ) 

    return model


def log_batch(image_batch, label_batch, classes):
    """Logs 16 labeled samples to Wandb"""
    examples = [wandb.Image(image_batch[n], caption=tf.boolean_mask(classes, label_batch[n])[0].numpy().decode()) for n in range(16)]
    wandb.log({"examples": examples})
    
    
def build_finetune_model(base_model, dropouts, fc_layers, activation, num_classes):
    """Freezes model body and intializes head with Wandb.config hyperparameters"""
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)

    for fc, drop in zip(fc_layers, dropouts):
        x = tf.keras.layers.Dense(fc, activation=activation)(x) 
        x = tf.keras.layers.Dropout(drop)(x)

    classification = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification')(x)

    return tf.keras.Model(inputs=base_model.input, outputs=classification)

    
def image_dataset_generator(uri, batch=16, shuffle=True):
    """Generates image batches from S3"""
    # https://stackoverflow.com/questions/41194726/python-generator-thread-safety-using-keras
    # https://anandology.com/blog/using-iterators-and-generators/
    file_gen = fs.walk(uri)
    classes = next(file_gen)[1]
    one_hot_encoder = dict(zip(classes, np.eye(len(classes))))
    dataset = [('s3://' + c[0] + '/' + f, one_hot_encoder[c[0].split('/')[-1]]) for c in file_gen for f in c[2]]

    i = 0

    while True:
                
        if shuffle and i == 0:
            np.random.shuffle(dataset)

        images, labels = [], []
        
        try:
            for sample in range(batch):
                path, label = dataset[i]

                with fs.open(path) as f:
                    img = f.read()
                
                img = tf.io.decode_image(img)
                img = tf.image.convert_image_dtype(img, tf.float32)
                images.append(img)

                label = tf.convert_to_tensor(label)
                labels.append(label)

                i += 1
        except IndexError:
            i = 0
            continue
            
        yield tf.convert_to_tensor(images), tf.convert_to_tensor(labels), [None] 

        
def _load_dataset(batch=16):
    path = 's3://ssa-data/dataset/source' # probably should parameterize
    train_ds = image_dataset_generator(f'{path}/training', batch=batch)
    val_ds = image_dataset_generator(f'{path}/validation', batch=batch)
    classes = [d['Key'].split('/')[-1] for d in fs.listdir(f'{path}/training') if d['StorageClass']=='DIRECTORY']
    num_train = sum([len(fs.ls(f'{path}/training/{c}')) for c in classes])
    num_val = sum([len(fs.ls(f'{path}/validation/{c}')) for c in classes])
    train_steps = int(np.ceil(num_train/batch))
    val_steps = int(np.ceil(num_val/batch))
    return train_ds, train_steps, val_ds, val_steps, classes
    
    
if __name__=="__main__":
    
    fs = s3fs.S3FileSystem()
    
    # TODO(developer): load the config from the best model
    config_defaults = {
        'learning_rate' : 1e-4,
        'epochs' : 10,
        'batch' : 16,
        'dropout' : 0.5,
        'optimizer' : 'adam',
        'activation' : 'relu',
        'hidden_ly1' : 256,
        'hidden_ly2' : 256,
        'dropout_ly1' : 0.1,
        'dropout_ly2' : 0.1,
        'model': 'ResNet-50'
    }
    
    wandb.init(config=config_defaults)
    
    train()
    