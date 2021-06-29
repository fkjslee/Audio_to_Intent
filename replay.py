import os
from preprocess.preprocessor import init_jieba
from AudioProcessor import AudioRecognizer
import sounddevice as sd
import time
from utils import init_logger
import soundfile as sf
import logging


if __name__ == "__main__":
    init_jieba()
    init_logger()
    logger = logging.getLogger(__name__)
    record_samplerate = 16000
    recognizer = AudioRecognizer(record_samplerate, replay=True)
    recognizer.start()

    audio_root_path = "replay"
    assert os.path.isdir(audio_root_path), "%s should be a dir, and include audio files which you want to replay" % audio_root_path
    for filename in os.listdir(audio_root_path):
        if not filename.endswith(".wav") and not filename.endswith(".mp3"):
            continue
        filepath = os.path.join(audio_root_path, filename)
        logger.info("replay: filepath = %s" % filepath)
        indata, samplerate = sf.read(filepath, dtype='int16')
        assert samplerate == record_samplerate
        recognizer.write(indata.tobytes())
        time.sleep(1)  # wait AudioRecognizer to process
