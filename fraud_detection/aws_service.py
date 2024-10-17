import os
import boto3
import traceback
import json
import sys
from fraud_detection import AWS_ACCESS_KEY_ID_ENV_KEY, AWS_SECRET_ACCESS_KEY_ENV_KEY, AWS_REGION_NAME   

class AWSService:
    def __init__(self):
        self.s3_resource = boto3.resource(
                "s3",
                aws_access_key_id=  os.getenv(
                    AWS_ACCESS_KEY_ID_ENV_KEY,),
                aws_secret_access_key =  os.getenv(
                    AWS_SECRET_ACCESS_KEY_ENV_KEY,
                ),
                region_name= os.getenv(AWS_REGION_NAME)
            )
    def download_file_from_s3(self, local_file_path, bucket_name, file_name):
        # Downloading zip file from s3
        try:
            s3_obj = self.s3_resource.Object(bucket_name, key=file_name)
            print("s3_obj created", s3_obj)
            print(os.path.join(local_file_path, file_name))
            s3_obj.download_file(os.path.join(local_file_path, file_name))
        except Exception as e:
            print("Unable to dowload the file from s3")
            print(e)
        
    # def upload_file_to_s3(self, bucket_name, model_path, model_name):
    #     self.s3_resource.Bucket(bucket_name).upload_file(Filename = model_path, Key = model_name)
    def upload_file_to_s3(self, local_file_path, bucket_name):
        try:
            # Create an S3 object
            s3_object = self.s3_resource.Object(bucket_name, f"{os.path.dirname(local_file_path)}/{local_file_path.split('/')[-1]}")
            
            # Upload the file
            s3_object.upload_file(local_file_path)
            print(f"File uploaded successfully to {bucket_name}/{os.path.dirname(local_file_path)}/{local_file_path.split('/')[-1]}")
        
        except FileNotFoundError:
            print("The file was not found")
        except Exception:
            print("Credentials not available")