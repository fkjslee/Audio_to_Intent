from preprocess.preprocessor import init_jieba
from AudioProcessor import AudioRecognizer, Record_data
import sounddevice as sd
import time
from utils import init_logger
import numpy as np


def callback(indata, frames, ts, status):
    if status:
        print(status)
    recognizer.write(indata.tobytes())
    Record_data.data = np.append(Record_data.data, indata)


if __name__ == "__main__":
    init_jieba()
    init_logger()
    samplerate = 16000
    recognizer = AudioRecognizer(samplerate)
    try:
        recognizer.start()
        with sd.InputStream(device=1, channels=1, dtype="int16", samplerate=samplerate, callback=callback):
            while 1:
                time.sleep(0.04)

    except KeyboardInterrupt:  # Exception as e:
        print("Recognizer was interrupt by keyboard!")
    finally:
        recognizer.stop()
