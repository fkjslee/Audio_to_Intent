# coding: utf-8
import os
from asr.asr import speech_recognizer
from preprocess.preprocessor import init_jieba, init_asr
from asr.common.credential import Credential
from datetime import datetime
import json
import jieba
import logging
from utils import get_args
from network.msgsender import MsgSender
from utils import set_seed, get_args
from trainer import Trainer
import numpy as np
from scipy.io import wavfile
import time


logger = logging.getLogger(__name__)


class Record_data:
    data = np.zeros(0).reshape(0, 1).astype(np.int16)


def formatText(text):
    logger.info("asr result: %s" % text)
    cut_text = jieba.lcut(text)
    space_cut_text = " ".join(cut_text)
    logger.info("jieba cut message: %s", space_cut_text)
    return space_cut_text


class AudioListener(speech_recognizer.SpeechRecognitionListener):
    def __init__(self, id, predictor: Trainer, msg_sender: MsgSender, samplerate, replay=False):
        self.id = id
        self.predictor = predictor
        self.msg_sender = msg_sender
        self.samplerate = samplerate
        self.replay = replay

    def record_data(self):
        now = time.time()
        str_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(now))
        str_time = str_time + " " + str(now)
        folder_path = "log/audio_log"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not self.replay:
            wavfile.write(os.path.join(folder_path, str_time + ".wav"), self.samplerate, Record_data.data)
        Record_data.data = np.zeros(0).reshape(0, 1).astype(np.int16)

    def on_recognition_start(self, response):
        logger.info("%s|OnRecognitionStart\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def on_sentence_begin(self, response):
        if len(Record_data.data) > self.samplerate:  # when sentence begin, some word has already been read
            Record_data.data = Record_data.data[len(Record_data.data) - self.samplerate:]
        logger.info("%s|OnRecognitionSentenceBegin" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def on_recognition_result_change(self, response):
        # print(response['result']['voice_text_str'])
        ...

    def on_sentence_end(self, response):
        logger.info("%s|OnRecognitionEnd\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.record_data()
        text = response['result']['voice_text_str']
        space_cut_text = formatText(text)
        intent_str, intent_logit, all_word_str, all_word_logit = self.predictor.predict_sentence(space_cut_text)
        print('intent type(sorted)', intent_str)
        print('intent possibility', intent_logit)
        print('slot type(sorted)', all_word_str)
        print('slot possibility', all_word_logit)
        slot_map = {}
        for word, word_type in zip(space_cut_text.split(' '), all_word_str):
            slot_map[word_type[0]] = word
        self.msg_sender.send_msg(intent_str[0], slot_map)

    def on_recognition_complete(self, response):
        logger.info("%s|OnRecognitionComplete\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def on_fail(self, response):
        rsp_str = json.dumps(response, ensure_ascii=False)
        logger.info("%s|OnFail,message %s\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rsp_str))


class AudioRecognizer(speech_recognizer.SpeechRecognizer):
    def __init__(self, samplerate, replay=False):
        asrMsg = init_asr()
        args = get_args()
        set_seed(args)
        msg_sender = MsgSender(addr=args.command_server_addr, port=args.command_server_port)
        listener = AudioListener(0, Trainer(), msg_sender, samplerate, replay)
        super().__init__(asrMsg['APPID'], Credential(asrMsg['SECRET_ID'], asrMsg['SECRET_KEY']),
                         asrMsg['ENGINE_MODEL_TYPE'], listener)
        self.set_filter_modal(1)
        self.set_filter_punc(1)
        self.set_filter_dirty(1)
        self.set_need_vad(1)
        self.set_voice_format(1)
        self.set_word_info(1)
        self.set_convert_num_mode(1)
