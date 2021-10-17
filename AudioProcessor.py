# coding: utf-8
import os
import sys
import time
import json
import jieba
import logging
import threading
import numpy as np
from asr.asr import speech_recognizer
from preprocess.preprocessor import init_jieba, init_asr
from asr.common.credential import Credential
from datetime import datetime, timedelta
from network.msgsender import MsgSender
from utils import set_seed, get_args
from trainer import Trainer
from scipy.io import wavfile


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
        self.args = get_args()
        self.asr_fps = 0
        self.asr_cnt = 0
        self.prediction_fps = 0
        self.prediction_cnt = 0
        self.fps_interval = timedelta(seconds=1)
        self.last_update_asr_fps_time = None
        self.last_update_pred_fps_time = None
        self.lock = threading.Lock()
        self.predict_text = ""
        self.is_predicting = False
        if self.args.gui:
            sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "gui"))
            from gui import run_FFT_analyzer
            self.gui_controller = run_FFT_analyzer.AudioController()
            thread = threading.Thread(target=run_FFT_analyzer.run_FFT_analyzer, kwargs={"controller": self.gui_controller})
            thread.start()

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

    def _predict(self, end):
        space_cut_text = formatText(self.predict_text)
        result = {}
        for model_name in self.args.predict_slots:
            result[model_name] = self.predictor.predict_sentence(space_cut_text, which_slot=model_name)
            logger.info('{} type(sorted): {}'.format(model_name, result[model_name][0]))
            logger.info('{} possibility: {}'.format(model_name, result[model_name][1]))
        if self.args.task == 'qiyuan':
            intent_str, intent_logit = result['intent']
            moved_object_str, moved_object_logit = result['B-moved_object']
            moved_position_str, moved_position_logit = result['B-moved_position']
            slot_map = {'B-moved_object': moved_object_str[0], 'B-moved_position': moved_position_str[0]}
            if end:
                self.msg_sender.send_msg(intent_str[0], slot_map)
        self.is_predicting = False

    def predict(self, end):
        if self.is_predicting and not end:
            return
        self.is_predicting = True

        threading.Thread(target=self._predict, kwargs={"end": end}).start()

        self.lock.acquire()
        if self.last_update_pred_fps_time is None:
            self.last_update_pred_fps_time = datetime.now()
            self.prediction_cnt = 1
        elif datetime.now() - self.last_update_pred_fps_time < self.fps_interval:
            self.prediction_cnt += 1
        else:
            self.prediction_fps = self.prediction_cnt / self.fps_interval.total_seconds()
            self.last_update_pred_fps_time = datetime.now()
            self.prediction_cnt = 0
        self.lock.release()

    def on_recognition_start(self, response):
        logger.info("%s|OnRecognitionStart\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def on_sentence_begin(self, response):
        if self.args.gui:
            self.gui_controller.hidden = False
        if len(Record_data.data) > self.samplerate:  # when sentence begin, some word has already been read
            Record_data.data = Record_data.data[len(Record_data.data) - self.samplerate:]
        logger.info("%s|OnRecognitionSentenceBegin" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def on_recognition_result_change(self, response):
        self.lock.acquire()
        if self.last_update_asr_fps_time is None:
            self.last_update_asr_fps_time = datetime.now()
            self.asr_cnt = 1
        elif datetime.now() - self.last_update_asr_fps_time < self.fps_interval:
            self.asr_cnt += 1
        else:
            self.asr_fps = self.asr_cnt / self.fps_interval.total_seconds()
            self.last_update_asr_fps_time = datetime.now()
            self.asr_cnt = 0
        self.predict_text = response['result']['voice_text_str']
        self.gui_controller.fps_msg = "ASR Fps: {} Predict Fps: {}".format(int(round(self.asr_fps)), int(round(self.prediction_fps)))
        self.lock.release()
        self.predict(end=False)

    def on_sentence_end(self, response):
        logger.info("%s|OnRecognitionEnd\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.record_data()
        self.lock.acquire()
        self.predict_text = response['result']['voice_text_str']
        self.gui_controller.fps_msg = "ASR Fps: {} Predict Fps: {}".format(int(round(self.asr_fps)), int(round(self.prediction_fps)))
        self.lock.release()
        self.predict(end=True)
        if self.args.gui:
            self.gui_controller.hidden = True

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
