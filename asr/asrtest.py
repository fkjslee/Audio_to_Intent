# -*- coding: utf-8 -*-
# 引用 SDK

import time
import sys
import threading
from datetime import datetime
import json

sys.path.append("../..")
from common import credential
from asr import speech_recognizer
import sounddevice as sd
import samplerate
import numpy as np
import wave
import base64
import jieba
from intentpredictor import IntentPredictor
import os
from utils import get_args
import yaml


# init asr
f = open(os.path.join("asrConfig.yml"), 'r', encoding='utf-8')
d = yaml.load(f.read(), yaml.FullLoader)
APPID = d['APPID']
SECRET_ID = d['SECRET_ID']
SECRET_KEY = d['SECRET_KEY']
ENGINE_MODEL_TYPE = d['ENGINE_MODEL_TYPE']
SLICE_SIZE = d['SLICE_SIZE']


class MySpeechRecognitionListener(speech_recognizer.SpeechRecognitionListener):
    def __init__(self, id, predictor):
        self.id = id
        self.predictor = predictor

    def on_recognition_start(self, response):
        print("%s|%s|OnRecognitionStart\n" % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), response['voice_id']))

    def on_sentence_begin(self, response):
        rsp_str = json.dumps(response, ensure_ascii=False)
        print("%s|%s|OnRecognitionSentenceBegin, rsp %s\n" % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), response['voice_id'], rsp_str))

    def on_recognition_result_change(self, response):
        # print(response['result']['voice_text_str'])
        ...

    def on_sentence_end(self, response):
        # rsp_str = json.dumps(response, ensure_ascii=False)
        text = response['result']['voice_text_str']
        print(" ".join(jieba.lcut(text)))
        print(self.predictor.predict(text))
        # print("%s|%s|OnSentenceEnd, rsp %s\n" % (datetime.now().strftime(
        #     "%Y-%m-%d %H:%M:%S"), response['voice_id'], rsp_str))

    def on_recognition_complete(self, response):
        print("%s|%s|OnRecognitionComplete\n" % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), response['voice_id']))

    def on_fail(self, response):
        rsp_str = json.dumps(response, ensure_ascii=False)
        print("%s|%s|OnFail,message %s\n" % (datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"), response['voice_id'], rsp_str))


def callback(indata, frames, time, status):
    if status:
        print(status)
    recognizer.write(indata.tobytes())


if __name__ == "__main__":
    listener = MySpeechRecognitionListener(0, IntentPredictor())
    credential_var = credential.Credential(SECRET_ID, SECRET_KEY)
    recognizer = speech_recognizer.SpeechRecognizer(
        APPID, credential_var, ENGINE_MODEL_TYPE, listener)
    recognizer.set_filter_modal(1)
    recognizer.set_filter_punc(1)
    recognizer.set_filter_dirty(1)
    recognizer.set_need_vad(1)
    recognizer.set_voice_format(1)
    recognizer.set_word_info(1)
    recognizer.set_convert_num_mode(1)

    try:
        recognizer.start()
        with sd.InputStream(device=1, channels=1, dtype="int16",
                            samplerate=16000, callback=callback):
            while 1:
                time.sleep(0.04)

    except KeyboardInterrupt:  # Exception as e:
        print(".....")
    finally:
        recognizer.stop()
