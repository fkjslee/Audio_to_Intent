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
from network.msgsender import MsgSender

logger = logging.getLogger(__name__)


class AudioListener(speech_recognizer.SpeechRecognitionListener):
    def __init__(self, id, predictor, msg_sender: MsgSender):
        self.id = id
        self.predictor = predictor
        self.msg_sender = msg_sender

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
        logger.info("asr result: %s" % text)
        cut_text = jieba.lcut(text)
        space_cut_text = " ".join(cut_text)
        logger.info("jieba cut message: %s", space_cut_text)
        intent_pred, slot_pred_list = self.predictor.predict(space_cut_text)
        intent_pred, slot_pred_list = intent_pred[0], slot_pred_list[0]
        logger.info("predict intent: %s\n predict slot: %s", str(get_intent_labels(get_args())[intent_pred]), str(slot_pred_list))
        slot_map = {}
        for word, entity in zip(cut_text, slot_pred_list):
            slot_map[entity] = word
        self.msg_sender.send_msg(str(get_intent_labels(get_args())[intent_pred]), slot_map)

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
        args = get_args()
        msg_sender = MsgSender(addr=args.command_server_addr, port=args.command_server_port)
        listener = AudioListener(0, IntentPredictor(), msg_sender)
        super().__init__(asrMsg['APPID'], Credential(asrMsg['SECRET_ID'], asrMsg['SECRET_KEY']),
                         asrMsg['ENGINE_MODEL_TYPE'], listener)
        self.set_filter_modal(1)
        self.set_filter_punc(1)
        self.set_filter_dirty(1)
        self.set_need_vad(1)
        self.set_voice_format(1)
        self.set_word_info(1)
        self.set_convert_num_mode(1)
