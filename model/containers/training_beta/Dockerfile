FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/tensorflow-training:2.3.0-gpu-py37-cu102-ubuntu18.04

WORKDIR /opt/training
COPY model.py .
COPY entrypoint.sh .

RUN pip install --upgrade pip -q
RUN pip install awscli pillow wandb boto3 -q

ENTRYPOINT ["./entrypoint.sh"]
