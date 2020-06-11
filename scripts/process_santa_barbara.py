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
from random import choice

logging.basicConfig(level=logging.NOTSET)
FLAGS = tf.compat.v1.app.flags.FLAGS

patterns = [r'(\.)+', r'(\([a-zA-Z0-9 _\(\)]+\))+', r'(<YWN)+', r'(?<=\s)=+\b', r'\b=+', r'\d(?=\w+)', r'(?<=\w)+\d', r'%(?=\w)+',
            r'\[(?=((\s|\'|\w|=|~|@|<|\-|\)|\()+))', r'(?<=(\s|\'|\w|=|~|@|>|\-|\)))+\]', r'(?<=\s)\-+', r'<[\w@%]+', r'[\w@%]+>', r'@+']


def main(_):
    source_dir = FLAGS.source_dir
    transcripts = glob(source_dir + "/transcripts/TRN/*.trn")
    clip_range = range(5, 14)
    df = pd.read_csv(transcripts[0], sep="\t", header=None)
    clip_duration = choice(clip_range)
    clip_transcript = ''
    curr_start = 0
    df_transcripts = pd.DataFrame(columns=["start", "end", "transcript", "selected clip_duration", "actual clip duration"])
    columns = df.columns
    df_index = 0
    for i in df.index:
        curr_transcript = df[columns[-1]][i]
        time_map = df[columns[0]][i]
        start, end = time_map.split(" ")
        start = float(start)
        end = float(end)
        curr_transcript = re.sub(r'(' + "|".join(patterns) + r')', '', curr_transcript)
        curr_transcript = re.sub(r'((?<=\s)=+\b|\b=+|,|\?|~)', '', curr_transcript)
        curr_transcript = re.sub(r'\s+', ' ', curr_transcript)
        curr_transcript = curr_transcript.strip()
        if re.sub('[^a-zA-Z]', '', curr_transcript) == '':
            curr_transcript = ''
        clip_transcript += " " + curr_transcript
        if end - curr_start > clip_duration:
            clip_transcript = re.sub(r'\s+', ' ', clip_transcript).strip()
            df_transcripts.loc[df_index] = [curr_start, end, clip_transcript, clip_duration, end - curr_start]
            df_index += 1
            curr_start = end
            clip_transcript = ''
            clip_duration = choice(clip_range)
    if clip_transcript != '':
        clip_transcript += curr_transcript
        df_transcripts.loc[df_index] = [curr_start, end, clip_transcript]
        clip_transcript = ''
    df_transcripts.to_csv(FLAGS.data_dir + "/transcripts.csv", index=False)


if __name__ == "__main__":
    tf.compat.v1.app.flags.DEFINE_string("source_dir", "source dir", "source directory")
    tf.compat.v1.app.flags.DEFINE_string("data_dir", "data dir", "data directory")
    tf.compat.v1.app.flags.DEFINE_float("test_size", 0.02, "test size")
    tf.compat.v1.app.flags.DEFINE_integer("random_state", 1254, "random state")
    tf.compat.v1.app.run()
