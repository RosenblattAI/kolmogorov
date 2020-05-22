# Downloads the TensorFlow library used to run the Python 3 script
FROM tensorflow/tensorflow:2.0.0-gpu-py3

# Contains the common functionality necessary to create a container compatible with SageMaker
RUN pip install sagemaker-containers -q 

# Wandb allows us to customize and centralize logging
RUN pip install wandb -q --upgrade

# Copies the training code inside the container according to the SageMaker Estimator design pattern 
COPY model.py /opt/ml/code/model.py
COPY callbacks.py /opt/ml/code/callbacks.py
COPY wandb_setup.sh /opt/ml/code/wandb_setup.sh

# Set the entry point as the setup script
ENV SAGEMAKER_PROGRAM wandb_setup.sh