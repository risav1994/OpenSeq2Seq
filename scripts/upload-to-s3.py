from requests import request
import logging
import tensorflow as tf
logging.basicConfig(level=logging.NOTSET)

FLAGS = tf.compat.v1.app.flags.FLAGS


def main(_):
    r = request("GET", "https://lab41openaudiocorpus.s3.amazonaws.com/VOiCES_release.tar.gz", stream=True)
    import boto3
    session = boto3.Session(aws_access_key_id=FLAGS.access_key,
                            aws_secret_access_key=FLAGS.secret_key, region_name="us-east-1")
    s3 = session.resource("s3")
    bucket_name = FLAGS.aws_bucket
    bucket = s3.Bucket(bucket_name)
    key = "gpu-server-data/deployer_ep/datasets/speech-to-text/voices/VOiCES_release.tar.gz"
    transfer_config = boto3.s3.transfer.TransferConfig(multipart_threshold=104857600, multipart_chunksize=104857600)
    bucket.upload_fileobj(r.raw, key, Config=transfer_config)


if __name__ == '__main__':
    tf.compat.v1.app.flags.DEFINE_string("access_key", "aws access key", "aws access key")
    tf.compat.v1.app.flags.DEFINE_string("secret_key", "aws secret key", "aws secret key")
    tf.compat.v1.app.flags.DEFINE_string("aws_bucket", "aws bucket", "aws bucket")
    tf.compat.v1.app.run()
