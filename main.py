import jieba
import yaml
from preprocess.preprocessor import init_jieba, init_asr
from AudioProcessor import AudioRecognizer
import sounddevice as sd
import time
import argparse
from utils import init_logger
from network.msgsender import MsgSender


def callback(indata, frames, time, status):
    if status:
        print(status)
    recognizer.write(indata.tobytes())


if __name__ == "__main__":
    init_jieba()
    init_logger()
    recognizer = AudioRecognizer()
    try:
        recognizer.start()
        with sd.InputStream(device=1, channels=1, dtype="int16", samplerate=16000, callback=callback):
            while 1:
                time.sleep(0.04)

    except KeyboardInterrupt:  # Exception as e:
        print("Recognizer was interrupt by keyboard!")
    finally:
        recognizer.stop()
