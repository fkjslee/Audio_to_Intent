# coding: utf-8
from asr.asr import speech_recognizer
from preprocess.preprocessor import init_jieba, init_asr
from asr.common.credential import Credential
from intentpredictor import IntentPredictor
from datetime import datetime
import json
import jieba
import logging
from utils import get_intent_labels, get_slot_labels, get_args

logger = logging.getLogger(__name__)


class AudioListener(speech_recognizer.SpeechRecognitionListener):
    def __init__(self, id, predictor):
        self.id = id
        self.predictor = predictor

    def on_recognition_start(self, response):
        logger.info("%s|OnRecognitionStart\n" % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def on_sentence_begin(self, response):
        logger.info("%s|OnRecognitionSentenceBegin" % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def on_recognition_result_change(self, response):
        # print(response['result']['voice_text_str'])
        ...

    def on_sentence_end(self, response):
        text = response['result']['voice_text_str']
        logger.info("asr msg: %s" % text)
        logger.info("msg been cut: %s", " ".join(jieba.lcut(text)))
        intent_preds, slot_preds_list = self.predictor.predict(text)
        logger.info("predict intent: %s", str(get_intent_labels(get_args())[intent_preds[0]]))
        logger.info("predict slot: %s", str(slot_preds_list[0]))

    def on_recognition_complete(self, response):
        logger.info("%s|OnRecognitionComplete\n" % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def on_fail(self, response):
        rsp_str = json.dumps(response, ensure_ascii=False)
        logger.info("%s|OnFail,message %s\n" % (datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"), rsp_str))


class AudioRecognizer(speech_recognizer.SpeechRecognizer):
    def __init__(self):
        asrMsg = init_asr()
        listener = AudioListener(0, IntentPredictor())
        super().__init__(asrMsg['APPID'], Credential(asrMsg['SECRET_ID'], asrMsg['SECRET_KEY']),
                         asrMsg['ENGINE_MODEL_TYPE'], listener)
        self.set_filter_modal(1)
        self.set_filter_punc(1)
        self.set_filter_dirty(1)
        self.set_need_vad(1)
        self.set_voice_format(1)
        self.set_word_info(1)
        self.set_convert_num_mode(1)
