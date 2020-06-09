import tensorflow as tf
import librosa
import pandas as pd
import os
import unicodedata
import soundfile
import logging
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.NOTSET)
FLAGS = tf.compat.v1.app.flags.FLAGS


def main(_):
    source_dir = FLAGS.source_dir
    transcripts = glob(source_dir + "/transcripts/TRN/*.trn")
    df = pd.read_csv(transcripts[0], sep="\t", header=None)
    print(df)


if __name__ == "__main__":
    tf.compat.v1.app.flags.DEFINE_string("source_dir", "source dir", "source directory")
    tf.compat.v1.app.flags.DEFINE_float("test_size", 0.02, "test size")
    tf.compat.v1.app.flags.DEFINE_integer("random_state", 1254, "random state")
    tf.compat.v1.app.run()
