import librosa
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from threading import Thread

FLAGS = tf.compat.v1.flags.FLAGS


def remove_silence(file, bar):
    y, sr = librosa.load(file)
    yt, index = librosa.effects.trim(y, top_db=10)
    librosa.output.write_wav(file, yt, sr)
    bar.update(1)


def main(_):
    data_dir = FLAGS.data_dir
    files = glob(data_dir + "/*.wav")
    threads = []
    with tqdm(total=len(files)) as bar:
        for file in files:
            thread = Thread(target=remove_silence, args=(file, bar,))
            threads.append(thread)
            thread.start()
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    tf.compat.v1.flags.DEFINE_string("data_dir", "data dir", "data directory")
    tf.compat.v1.app.run()
