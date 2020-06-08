from sox import Transformer
from glob import glob
import tensorflow as tf

FLAGS = tf.compat.v1.app.flags.FLAGS


def main(_):
    print(FLAGS)


if __name__ == '__main__':
    tf.compat.v1.app.flags.DEFINE_string("data-dir", "data dir", "data directory")
    tf.compat.v1.app.run()
