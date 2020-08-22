import tensorflow as tf

class Model:
    def __init__(self, config, num_classes=18, input_shape=(224,224,3), translation=True):
        
        self.input_shape = input_shape
        self.translation = translation
        self.num_classes = num_classes
        
        for k,v in config.items():
            setattr(self, k, v['value'] if type(v)==dict else v)
            
        try:
            model_cl = getattr(tf.keras.applications, self.model)
        except AttributeError:
            raise SyntaxError(f'Model "{self.model}" was not found, please adjust your sweep configuration.')

        # get pretrained model with top removed
        model = model_cl(
            include_top=False, 
            weights='imagenet' if self.imagenet else None, 
            input_shape=self.input_shape
        )
        
        model = self.add_preprocessing(model)
        self.tf_model = self.build_model(model)
        

        
    def add_preprocessing(self, base_model):
        """ Adds a RandomTranslation layer to the front of the model"""
        preprocessing_lys = [tf.keras.layers.experimental.preprocessing.Normalization()]
        if self.translation:
            preprocessing_lys.insert(
                0,
                tf.keras.layers.experimental.preprocessing.RandomTranslation(
                    height_factor=0.4, 
                    width_factor=0.4, 
                    fill_mode='constant'
                )
            )

        model  = tf.keras.models.Sequential([base_model.input] + preprocessing_lys + [base_model])
        return model
    
    
    def build_model(self, base_model):
        """"Takes model body and intializes head with Wandb.config hyperparameters"""
        output_lys = []

        x = base_model.output

        try:
            base_out_ly = getattr(tf.keras.layers, self.base_output_setting)
        except AttributeError:
            raise SyntaxError(f'Layer "{self.base_output_setting}" was not found, please adjust your sweep configuration.')

        # TODO: add check for activation_fn
        try:
            activation_fn = getattr(tf.keras.activations, self.activation)
        except AttributeError:
            raise SyntaxError(f'Activation function "{self.base_output_setting}" was not found, please adjust your sweep configuration.')

        x = base_out_ly()(x)
        c = tf.keras.layers.Dense(self.hidden_classification_ly, activation=self.activation )(x) 
        c = tf.keras.layers.Dropout(self.dropout_classification_ly )(c)
        classification = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='classification')(c)
        output_lys.append(classification)

        if self.orientation:
            o = tf.keras.layers.Dense(self.hidden_orientation_ly, activation=self.activation)(x) 
            o = tf.keras.layers.Dropout(self.dropout_orientation_ly)(o)
            orientation = tf.keras.layers.Dense(3, activation='sigmoid', name='orientation')(o) # TODO: remember to fix!
            output_lys.append(orientation)

        return tf.keras.Model(inputs=base_model.input, outputs=output_lys)