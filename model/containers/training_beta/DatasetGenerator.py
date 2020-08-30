import os
import numpy as np
import tensorflow as tf
from pathlib import Path

class DatasetGenerator(object):
    
    def __init__(self, path, multitask=False, distortion=False, distance=False, evaluate=False, batch=16, shuffle=True):
        self.batch = batch
        self.shuffle = shuffle
        self.evaluate = evaluate
        self.multitask = multitask
        self.distortion = distortion
        self.distance = distance
        self.path = Path(path).resolve()
        self.classes = sorted(next(os.walk(self.path))[1])
        
        try:
            self.samples = sum([len(next(os.walk(f'{self.path}/{c}'))[2]) for c in self.classes])
        except Exception as e:
            # TODO: Custom error here
            raise RuntimeError(f'DatasetGenerator: There was an exception getting the number of training and validation samples (are they all downloaded?): {str(e)}')

        self.steps = int(np.ceil(self.samples/self.batch))
    
    
    def _extract_orientation(self, filename):
        """Extracts the x,y,z floating point orientation values from a filename"""
        o_string = Path(filename).name.split('_')[2]
        return [float(val) for val in o_string.split(',')]
    
    
    def _extract_distortion(self, filename):
        """Extracts the aperture and fried parameter values from a filename"""
        distortion_metrics = filename.split('_')[4:6]
        try:
            D = float(distortion_metrics[0])
            r0 = float(distortion_metrics[1])
            return D/r0
        except ValueError:
            return 0.0


    def _extract_distance(sef, filename):
        """Extacts the Unity Distance value from a filename"""
        idx = filename.rindex(".",0,-4)
        distance_str = filename[:idx].split('_')[3]
        return float(distance_str)
    
    
    def __iter__(self):
        """Generates dataset batches"""
        file_gen = os.walk(self.path)
        _ = next(file_gen)[1]
        one_hot_encoder = dict(zip(self.classes, np.eye(len(self.classes))))
        dataset = []
        for class_ in file_gen:
            for filename in class_[2]:
                try:
                    dataset.append([
                            f'{class_[0]}/{filename}',
                            one_hot_encoder[class_[0].split('/')[-1]]
                        ]
                    )
                except:
                    print(f'failed to parse {class_}')
#         dataset = [[f'{class_[0]}/{filename}', one_hot_encoder[class_[0].split('/')[-1]]] for class_ in file_gen for filename in class_[2]]

        i = 0

        while True:

            if self.shuffle and i == 0:
                np.random.shuffle(dataset)

            if self.multitask:
                orients = []
                
            if self.distortion:
                distorts = []

            if self.distance:
                distances = []

            images, one_hots = [], []

            try:
                for sample in range(self.batch):
                    
                    path, one_hot = dataset[i*self.batch + sample]
                    with open(path, 'rb') as img:
                        f = img.read()
                        b_img = bytes(f)

                    img = tf.io.decode_image(b_img)
                    img = tf.image.convert_image_dtype(img, tf.float32)
                    images.append(img)

                    one_hot = tf.convert_to_tensor(one_hot)
                    one_hots.append(one_hot)
                    
                    if self.multitask:
                        if self.evaluate:
                            orient = tf.convert_to_tensor(self._extract_orientation(path))
                        else:
                            orient = tf.convert_to_tensor(self._extract_orientation(path))
                            orient = np.deg2rad(orient)
                            orient = tf.concat([tf.math.sin(orient), tf.math.cos(orient)], 0)
                        orients.append(orient)

                    if self.distortion:
                        distort = tf.convert_to_tensor(self._extract_distortion(path))
                        distorts.append(distort)

                    if self.distance:
                        distance = tf.convert_to_tensor(self._extract_distance(path))
                        distances.append(distance)


                    i += 1
            except IndexError:
                i = 0
                continue
            
            inputs = tf.convert_to_tensor(images)
            outputs = [tf.convert_to_tensor(one_hots)]
            
            if self.multitask:
                outputs.append(tf.convert_to_tensor(orients))
            if self.distortion:
                outputs.append(tf.convert_to_tensor(distorts))
            if self.distance:
                outputs.append(tf.convert_to_tensor(distances))
            
            yield inputs, outputs