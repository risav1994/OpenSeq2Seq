import tensorflow as tf
import librosa
import pandas as pd
import os
import unicodedata
import soundfile
import logging
import re
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.NOTSET)
FLAGS = tf.compat.v1.app.flags.FLAGS

patterns = [r'(\.)+', r'(\([a-zA-Z0-9 \(\)]+\))+', r'(<YWN)+', r'(?<=\s)=+\b', r'\b=+', r'\d(?=\w+)', r'(?<=\w)+\d', r'%(?=\w)+',
            r'\[(?=((\s|\'|\w|=|~|@|<|\-|\)|\()+))', r'(?<=(\s|\'|\w|=|~|@|>|\-|\)))+\]', r'(?<=\s)\-+', r'<[\w@%]+', r'[\w@%]+>', r'@+']


def main(_):
    source_dir = FLAGS.source_dir
    transcripts = glob(source_dir + "/transcripts/TRN/*.trn")
    df = pd.read_csv(transcripts[0], sep="\t", header=None)
    df_transcripts = pd.DataFrame(columns=["time_map", "cleaned", "original"])
    columns = df.columns
    df_index = 0
    for i in df.index:
        curr_transcript = df[columns[-1]][i]
        time_map = df[columns[0]][i]
        curr_transcript = re.sub(r'(' + "|".join(patterns) + r')', '', curr_transcript)
        curr_transcript = re.sub(r'((?<=\s)=+\b|\b=+|,|\?)', '', curr_transcript)
        curr_transcript = re.sub(r'\s+', ' ', curr_transcript)
        curr_transcript = curr_transcript.strip()
        if curr_transcript == '' or re.sub('[^a-zA-Z]', '', curr_transcript) == '':
            continue
        df_transcripts.loc[df_index] = [time_map, curr_transcript, df[columns[-1]][i]]
        df_index += 1
    df_transcripts.to_csv(FLAGS.data_dir + "/transcripts.csv", index=False)


if __name__ == "__main__":
    tf.compat.v1.app.flags.DEFINE_string("source_dir", "source dir", "source directory")
    tf.compat.v1.app.flags.DEFINE_string("data_dir", "data dir", "data directory")
    tf.compat.v1.app.flags.DEFINE_float("test_size", 0.02, "test size")
    tf.compat.v1.app.flags.DEFINE_integer("random_state", 1254, "random state")
    tf.compat.v1.app.run()
