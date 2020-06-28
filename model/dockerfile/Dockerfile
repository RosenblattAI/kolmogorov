# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM tensorflow/tensorflow-training:2.2.0-gpu-py37-cu102-ubuntu18.04

RUN pip install wandb -q --upgrade

RUN wandb login $api_key

ENTRYPOINT wandb agent $sweep_id