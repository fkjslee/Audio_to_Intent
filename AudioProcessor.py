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
import pyttsx3
from xpinyin import Pinyin

logger = logging.getLogger(__name__)


class Record_data:
    data = np.zeros(0).reshape(0, 1).astype(np.int16)


def get_str_distance(l1, l2):
    dp = [[x + y for y in range(len(l1) + 1)] for x in range(len(l2) + 1)]
    for i in range(1, len(l2) + 1):
        for j in range(1, len(l1) + 1):
            if l2[i - 1] == l1[j - 1]:
                d = 0
            else:
                d = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + d)
    return dp[len(l2)][len(l1)]


def formatText(text, end):
    if end:
        logger.info("asr result: %s" % text)
    cut_text = jieba.lcut(text)
    space_cut_text = " ".join(cut_text)
    if end:
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
        self.asr_res = []
        self.pred_res = []
        self.all_other_res = []
        self.predict_text = ""
        self.is_predicting = False
        self.gui_lock = threading.Lock()
        self.have_seen = []
        self.have_seen_modify = []
        self.have_seen_lock = threading.Lock()
        if self.args.gui:
            sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "gui"))
            from gui import run_FFT_analyzer
            self.gui_controller = run_FFT_analyzer.AudioController()
            thread = threading.Thread(target=run_FFT_analyzer.run_FFT_analyzer,
                                      kwargs={"controller": self.gui_controller})
            thread.start()
        if self.args.manual_input:
            while True:
                self.predict_text = input("input sentence")
                self.predict(end=True)


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

    def emit_gui_res(self):
        self.gui_lock.acquire()
        self.gui_controller.pred_res = self.asr_res + self.pred_res
        self.gui_lock.release()

    def _predict(self, end):
        space_cut_text = formatText(self.predict_text, end)
        result = {}
        self.pred_res = []
        for model_name in self.args.predict_slots:
            result[model_name] = self.predictor.predict_sentence(space_cut_text, which_slot=model_name)
            if end:
                if model_name == "slot":
                    for word, slot_str, possibility in zip(space_cut_text.split(' '), result[model_name][0],
                                                           result[model_name][1]):
                        logging.info(
                            'predict {} as {} with possibility: {:.1f}%'.format(word, slot_str[0], possibility[0]))
                else:
                    logger.info('{} type(sorted): {}'.format(model_name, result[model_name][0]))
                    logger.info('{} possibility: {}'.format(model_name, result[model_name][1]))
            single_res = model_name + ": "
            if model_name == "slot":
                for word, slot_str, possibility in zip(space_cut_text.split(' '), result[model_name][0],
                                                       result[model_name][1]):
                    single_res += '{}:{}({:.1f}%)'.format(word, slot_str[0], possibility[0] * 100)
            else:
                for i in range(min(len(result[model_name][0]), 3)):
                    single_res += '{}({:.1f}%) '.format(result[model_name][0][i], result[model_name][1][i] * 100)
            self.pred_res.append(single_res)
            self.emit_gui_res()
        if self.args.task == "VGA":
            intent_str, intent_logit = result['intent']
            slot_str, slot_logit = result['slot']
            intent_str = intent_str[0]
            slot_str = [s[0] for s in slot_str]

            def get_sentence(slots, words):
                sentence = ""
                for word, s in zip(words, slots):
                    if s.find("sentence") != -1:
                        sentence += word
                return sentence

            def infer_char_from_word(src_word, dst_char):
                if src_word is None and dst_char is None:
                    return None
                elif src_word is None:
                    return Pinyin().get_pinyin(dst_char).split('-')[0]
                elif dst_char is None:
                    return None
                res = ""
                min_dis = 10000
                pinyin_src_word = Pinyin().get_pinyin(src_word).split('-')
                pinyin_dst_char = Pinyin().get_pinyin(dst_char).split('-')[0]
                for src_char, pinyin_src_char in zip(src_word, pinyin_src_word):
                    new_dis = get_str_distance(pinyin_dst_char, pinyin_src_char)
                    if min_dis > new_dis:
                        res = src_char
                        min_dis = new_dis
                return res


            def get_modify_char(slots, words):
                s_wrong_word, s_wrong_char, s_right_word, s_right_char = None, None, None, None
                for word, s in zip(words, slots):
                    if s == "S-wrong-word":
                        s_wrong_word = word
                    elif s == "S-wrong-char":
                        s_wrong_char = word
                    elif s == "S-right-word":
                        s_right_word = word
                    elif s == "S-right-char":
                        s_right_char = word
                s_wrong_char = infer_char_from_word(s_wrong_word, s_wrong_char)
                s_right_char = infer_char_from_word(s_right_word, s_right_char)
                return s_wrong_char, s_right_char

            def get_modify_word(slots, words):
                s_wrong_word, s_right_word = None, None
                for word, s in zip(words, slots):
                    if s == "S-wrong-word":
                        s_wrong_word = word
                    elif s == "S-right-word":
                        s_right_word = word
                return s_wrong_word, s_right_word

            if intent_str == "add_sentence":
                sentence = get_sentence(slot_str, space_cut_text.split(" "))
                if end:
                    try:
                        self.have_seen_lock.acquire()
                        need_trans = False
                        replace_sentence = None
                        for seen_sentence, modify_sentence in zip(self.have_seen, self.have_seen_modify):
                            sentence_dis = get_str_distance(Pinyin().get_pinyin(sentence).split('-'),
                                                            seen_sentence)
                            if sentence_dis < len(seen_sentence) * 0.2:
                                need_trans = True
                            if sentence_dis == 0:
                                replace_sentence = modify_sentence
                                break
                        self.have_seen_lock.release()
                        if replace_sentence is not None:
                            sentence = replace_sentence
                        elif need_trans:
                            sender = MsgSender("166.111.139.25", 9030)
                            self.predict_text = json.loads(sender.send_msg(" ".join(list(self.predict_text))))[
                                'sentence']
                            self.predict_text = self.predict_text.replace(" ", "")
                            print("receive translated msg", self.predict_text)
                    except Exception as e:
                        print("error when send msg to translation", e)
                    msg = json.dumps({"intent": intent_str, "sentence": sentence, "confidence": str(intent_logit[0])})
                    result = self.msg_sender.send_msg(msg)
                    if result.find("need confirm") != -1:
                        engine = pyttsx3.init()
                        say_content = "确认" + ("删除" if intent_str == "delete_sentence" else "添加") + sentence
                        engine.say(say_content)

            if intent_str == "delete_sentence":
                sentence = get_sentence(slot_str, space_cut_text.split(" "))
                msg = json.dumps({"intent": intent_str, "sentence": sentence, "confidence": str(intent_logit[0])})
                if end:
                    result = self.msg_sender.send_msg(msg)
                    if result.find("need confirm") != -1:
                        engine = pyttsx3.init()
                        say_content = "确认" + ("删除" if intent_str == "delete_sentence" else "添加") + sentence
                        engine.say(say_content)
            elif intent_str == "modify_char":
                wrong_char, right_char = get_modify_char(slot_str, space_cut_text.split(" "))
                msg = json.dumps({"intent": intent_str, "S-wrong-char": wrong_char, "S-right-char": right_char, "confidence": str(intent_logit[0])})
                if end:
                    result = self.msg_sender.send_msg(msg)
                    sentence1 = json.loads(result).get("sentence1")
                    sentence2 = json.loads(result).get("sentence2")
                    if sentence1 is not None and sentence2 is not None:
                        self.have_seen_lock.acquire()
                        self.have_seen.append(str(Pinyin().get_pinyin(sentence1)).split('-'))
                        self.have_seen_modify.append(sentence2)
                        self.have_seen_lock.release()
                    if result.find("need confirm") != -1:
                        engine = pyttsx3.init()
                        say_content = "确认" + "把" + wrong_char + "改成" + right_char + "吗"
                        engine.say(say_content)

            elif intent_str == "modify_word":
                wrong_word, right_word = get_modify_word(slot_str, space_cut_text.split(" "))
                msg = json.dumps({"intent": intent_str, "S-wrong-word": wrong_word, "S-right-word": right_word, "confidence": str(intent_logit[0])})
                if end:
                    result = self.msg_sender.send_msg(msg)
                    sentence1 = json.loads(result).get("sentence1")
                    sentence2 = json.loads(result).get("sentence2")
                    if sentence1 is not None:
                        self.have_seen_lock.acquire()
                        self.have_seen.append(str(Pinyin().get_pinyin(sentence1)).split('-'))
                        self.have_seen_modify.append(sentence2)
                        self.have_seen_lock.release()
                    if result.find("need confirm") != -1:
                        engine = pyttsx3.init()
                        say_content = "确认" + "把" + wrong_word + "改成" + right_word + "吗"
                        engine.say(say_content)
            elif intent_str == "append_word":
                wrong_word, right_word = get_modify_word(slot_str, space_cut_text.split(" "))
                msg = json.dumps({"intent": intent_str, "S-wrong-word": wrong_word, "S-right-word": right_word, "confidence": str(intent_logit[0])})
                if end:
                    result = self.msg_sender.send_msg(msg)
                    sentence1 = json.loads(result).get("sentence1")
                    sentence2 = json.loads(result).get("sentence2")
                    if sentence1 is not None:
                        self.have_seen_lock.acquire()
                        self.have_seen.append(str(Pinyin().get_pinyin(sentence1)).split('-'))
                        self.have_seen_modify.append(sentence2)
                        self.have_seen_lock.release()
                    if result.find("need confirm") != -1:
                        engine = pyttsx3.init()
                        say_content = "确认" + "把" + wrong_word + "改成" + right_word + "吗"
                        engine.say(say_content)


        self.is_predicting = False

    def predict(self, end):
        if self.is_predicting or not end:
            return
        while self.is_predicting:  # wait until another thread finished its prediction, without this two thread may predict simultaneously
            time.sleep(0.1)
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
        if self.args.manual_input:
            return
        self.asr_res = ["ASR result: " + response['result']['voice_text_str']]
        self.emit_gui_res()
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
        self.gui_controller.fps_msg = "ASR Fps: {} Predict Fps: {}".format(int(round(self.asr_fps)),
                                                                           int(round(self.prediction_fps)))
        self.lock.release()
        self.predict(end=False)

    def on_sentence_end(self, response):
        if self.args.manual_input:
            return
        self.asr_res = ["ASR result: " + response['result']['voice_text_str']]
        logger.info("%s|OnRecognitionEnd\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.record_data()
        self.lock.acquire()
        self.predict_text = response['result']['voice_text_str']
        self.gui_controller.fps_msg = "ASR Fps: {} Predict Fps: {}".format(int(round(self.asr_fps)),
                                                                           int(round(self.prediction_fps)))
        self.lock.release()
        self.predict(end=True)
        # if self.args.gui:
        #     self.gui_controller.hidden = True

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
