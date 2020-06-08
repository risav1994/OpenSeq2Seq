from sox import Transformer
from glob import glob
import tensorflow as tf
import pandas as pd
import os
import unicodedata
from sklearn.model_selection import train_test_split

FLAGS = tf.compat.v1.app.flags.FLAGS


def main(_):
    source_dir = FLAGS.source_dir
    data = []
    df_details = pd.read_csv(os.path.join(source_dir, "validated.tsv"), sep="\t", header=0)
    for i in df_details.index:
        file_name = df_details["path"][i]
        source_file = os.path.join(source_dir, "clips/" + file_name)
        wav_file = os.path.join(os.path.dirname(__file__), "../data/common-voice-mozilla/Common-Voice-Mozilla/" + file_name.split(".mp3")[0] + ".wav")
        transcript = df_details["sentence"][i]
        transcript = unicodedata.normalize("NFKD", transcript) \
            .encode("ascii", "ignore")   \
            .decode("ascii", "ignore")

        transcript = transcript.lower().strip()
        if not os.path.exists(wav_file):
            Transformer().build(source_file, wav_file)
        wav_filesize = os.path.getsize(wav_file)
        data.append((os.path.abspath(wav_file), wav_filesize, transcript))
    train_data, test_data = train_test_split(data, test_size=FLAGS.test_size, random_state=FLAGS.random_state)
    df_train = pd.DataFrame(data=train_data, columns=["wav_filename", "wav_filesize", "transcript"])
    df_train.to_csv(os.path.join(os.path.dirname(__file__), "../data/common-voice-mozilla/train.csv", index=False))
    df_test = pd.DataFrame(data=test_data, columns=["wav_filename", "wav_filesize", "transcript"])
    df_test.to_csv(os.path.join(os.path.dirname(__file__), "../data/common-voice-mozilla/test.csv", index=False))


if __name__ == '__main__':
    tf.compat.v1.app.flags.DEFINE_string("source_dir", "source dir", "source directory")
    tf.compat.v1.app.flags.DEFINE_float("test_size", 0.02, "test size")
    tf.compat.v1.app.flags.DEFINE_integer("random_state", 1254, "random state")
    tf.compat.v1.app.run()
