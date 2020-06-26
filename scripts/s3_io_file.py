from boto3 import Session
from botocore.config import Config


class S3IOFile(object):
    def __init__(self, s3_path, aws_access_key_id, aws_secret_access_key, bucket, region_name="us-east-1"):
        self.session = Session(aws_access_key_id=aws_access_key_id,
                               aws_secret_access_key=aws_secret_access_key,
                               region_name=region_name)
        self.s3_client = self.session.client("s3", config=Config(signature_version='s3v4', s3={"use_accelerate_endpoint": True}))
        self.s3_path = s3_path
        self.bucket = bucket
        response = self.s3_client.head_object(Bucket=self.bucket, Key=self.s3_path)
        self.content_size = response['ContentLength']
        self.tar_obj = self.s3_client.get_object(Bucket=self.bucket, Key=self.s3_path)
        self.seek_position = 0

    def seek(self, *args, **kwargs):
        self.seek_position = args[0]

    def read(self, buf_size):
        return self.tar_obj["Body"].read(buf_size)
