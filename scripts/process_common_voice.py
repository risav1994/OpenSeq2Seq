from sox import Transformer
import sox
from glob import glob
import tensorflow as tf
import pandas as pd
import os
import unicodedata
import tqdm
import logging
import librosa
import numpy as np
import soundfile
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.NOTSET)
logging.getLogger('sox').setLevel(logging.ERROR)
FLAGS = tf.compat.v1.app.flags.FLAGS
tfm = Transformer()
tfm.set_output_format(rate=16000)


def main(_):
    source_dir = FLAGS.source_dir
    data = []
    df_details = pd.read_csv(os.path.join(source_dir, "validated.tsv"), sep="\t", header=0)
    with tqdm.tqdm(total=len(df_details.index)) as bar:
        for i in df_details.index:
            file_name = df_details["path"][i]
            source_file = os.path.join(source_dir, "clips/" + file_name)
            wav_file = os.path.join(os.path.dirname(__file__), "../data/common-voice-mozilla/Common-Voice-Mozilla/wav-files/" +
                                    file_name.split(".mp3")[0] + ".wav")
            transcript = df_details["sentence"][i]
            if pd.isnull(transcript):
                continue
            transcript = unicodedata.normalize("NFKD", transcript) \
                .encode("ascii", "ignore")   \
                .decode("ascii", "ignore")

            transcript = transcript.lower().strip()
            try:
                if not os.path.exists(wav_file):
                    tfm.build(source_file, wav_file)
                    # y, sr = soundfile.read(wav_file, dtype=np.float32)
                    y, sr = librosa.load(wav_file, sr=None)
                    yt, index = librosa.effects.trim(y, top_db=10)
                    yt = y[max(index[0] - 40000, 0): min(index[1] + 40000, len(y))]
                    soundfile.write(wav_file, yt, sr)
                    bar.update(1)
                wav_filesize = os.path.getsize(wav_file)
                data.append((os.path.abspath(wav_file), wav_filesize, transcript))
            except sox.core.SoxError:
                logging.info(f"Error in file: {source_file}")
            bar.update(1)

    train_data, test_data = train_test_split(data, test_size=FLAGS.test_size, random_state=FLAGS.random_state)
    df_train = pd.DataFrame(data=train_data, columns=["wav_filename", "wav_filesize", "transcript"])
    df_train.to_csv(os.path.join(os.path.dirname(__file__), "../data/common-voice-mozilla/train.csv"), index=False)
    df_test = pd.DataFrame(data=test_data, columns=["wav_filename", "wav_filesize", "transcript"])
    df_test.to_csv(os.path.join(os.path.dirname(__file__), "../data/common-voice-mozilla/test.csv"), index=False)


if __name__ == '__main__':
    tf.compat.v1.app.flags.DEFINE_string("source_dir", "source dir", "source directory")
    tf.compat.v1.app.flags.DEFINE_float("test_size", 0.02, "test size")
    tf.compat.v1.app.flags.DEFINE_integer("random_state", 1254, "random state")
    tf.compat.v1.app.run()
