import boto3
import wandb
import tensorflow as tf
    
class CurriculumCallback(tf.keras.callbacks.Callback):
        
    def __init__(self):
        self.threshold = 0.95
        self.variance = 0.1
        self.batch = boto3.client('batch')
        
    # log validation batch metrics
    def on_test_batch_end(self, batch, logs=None):
        
        if logs['accuracy'] >= self.threshold :
            
            self.variance += 0.1

            response = self.batch.submit_job(jobName='next-curriculum', # use your HutchNet ID instead of 'jdoe'
                                        jobQueue='mixed', # sufficient for most jobs
                                        jobDefinition='generate-curriculum', # use a real job definition
                                        containerOverrides={
                                            "environment": [
                                                {"name": "VARIANCE", "value": self.variance}
                                            ]
                                        })
    
    # log validation epoch final iteration
    def on_test_epoch_end(self, epoch, logs=None):
        
        if logs['accuracy'] >= self.threshold :
            
            # grab new curriculum
        