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
    clip_range = range(5, 10)
    data = []
    file_prefix = "sbc-"
    file_index = 1
    with tqdm(total=len(transcripts)) as bar:
        for idx, transcript_file in enumerate(transcripts):
            print(transcript_file)
            df = open(transcript_file, "r")
            clip_duration = choice(clip_range)
            clip_transcript = ''
            curr_start = 0
            curr_start_idx = 0
            df_transcripts = pd.DataFrame(columns=["start", "end", "transcript", "selected clip_duration", "actual clip duration"])
            df_index = 0
            audio_file = source_dir + "/clips/" + transcript_file.split("/")[-1].split(".trn")[0] + ".wav"
            audio_data, sr = librosa.load(audio_file, sr=None)
            audio_duration = librosa.get_duration(audio_data)
            for i, line in enumerate(df):
                line = line.replace("\n", "")
                curr_transcript = line.split("\t")[-1]
                time_map = re.split(r'(\t|\s)+', line)
                start, end = time_map[0], time_map[2]
                start = float(start)
                end = float(end)
                if pd.isnull(curr_transcript):
                    curr_transcript = ''

                curr_transcript = re.sub(r'(' + "|".join(patterns) + r')', '', curr_transcript)
                curr_transcript = re.sub(r'((?<=\s)=+\b|\b=+|,|\?|~)', '', curr_transcript)
                curr_transcript = re.sub(r'\s+', ' ', curr_transcript)
                curr_transcript = curr_transcript.strip()
                if re.sub('[^a-zA-Z]', '', curr_transcript) == '':
                    curr_transcript = ''
                clip_transcript += " " + curr_transcript
                if end - curr_start > clip_duration:
                    clip_transcript = re.sub(r'\s+', ' ', clip_transcript).strip()
                    clip_transcript = unicodedata.normalize("NFKD", clip_transcript) \
                        .encode("ascii", "ignore")   \
                        .decode("ascii", "ignore")
                    df_transcripts.loc[df_index] = [curr_start, end, clip_transcript, clip_duration, end - curr_start]
                    end_idx = int(end * sr)
                    curr_audio_data = audio_data[curr_start_idx: end_idx]
                    yt, index = librosa.effects.trim(curr_audio_data, top_db=10)
                    yt = curr_audio_data[max(index[0] - 40000, 0): min(index[1] + 40000, len(curr_audio_data))]
                    wav_file = FLAGS.data_dir + "/wav-files/" + file_prefix + str(file_index) + ".wav"
                    soundfile.write(wav_file, yt, sr)
                    file_index += 1
                    wav_filesize = os.path.getsize(wav_file)
                    data.append((os.path.abspath(wav_file), wav_filesize, clip_transcript))
                    df_index += 1
                    curr_start = end
                    curr_start_idx = end_idx
                    clip_transcript = ''
                    clip_duration = choice(clip_range)
            if clip_transcript != '':
                clip_transcript = re.sub(r'\s+', ' ', clip_transcript).strip()
                clip_transcript = unicodedata.normalize("NFKD", clip_transcript) \
                    .encode("ascii", "ignore")   \
                    .decode("ascii", "ignore")
                df_transcripts.loc[df_index] = [curr_start, end, clip_transcript, clip_duration, end - curr_start]
                curr_audio_data = audio_data[curr_start_idx:]
                yt, index = librosa.effects.trim(curr_audio_data, top_db=10)
                yt = curr_audio_data[max(index[0] - 40000, 0): min(index[1] + 40000, len(curr_audio_data))]
                wav_file = FLAGS.data_dir + "/wav-files/" + file_prefix + str(file_index) + ".wav"
                soundfile.write(wav_file, yt, sr)
                file_index += 1
                wav_filesize = os.path.getsize(wav_file)
                data.append((os.path.abspath(wav_file), wav_filesize, clip_transcript))
                clip_transcript = ''
            df_transcripts.to_csv(FLAGS.data_dir + "/transcripts-" + str(idx) + ".csv", index=False)
            bar.update(1)
    pd.DataFrame(data=data, columns=["wav_filename", "wav_filesize", "transcript"]).to_csv(FLAGS.data_dir + "/train.csv", index=False)


if __name__ == "__main__":
    tf.compat.v1.app.flags.DEFINE_string("source_dir", "source dir", "source directory")
    tf.compat.v1.app.flags.DEFINE_string("data_dir", "data dir", "data directory")
    tf.compat.v1.app.flags.DEFINE_float("test_size", 0.02, "test size")
    tf.compat.v1.app.flags.DEFINE_integer("random_state", 1254, "random state")
    tf.compat.v1.app.run()
