from asr.asr import speech_recognizer
from preprocess.preprocessor import init_jieba, init_asr
from asr.common.credential import Credential
from intentpredictor import IntentPredictor
from datetime import datetime
import json
import jieba


class AudioListener(speech_recognizer.SpeechRecognitionListener):
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


class AudioRecognizer(speech_recognizer.SpeechRecognizer):
    def __init__(self):
        asrMsg = init_asr()
        listener = AudioListener(0, IntentPredictor())
        super().__init__(asrMsg['APPID'], Credential(asrMsg['SECRET_ID'], asrMsg['SECRET_KEY']), asrMsg['ENGINE_MODEL_TYPE'], listener)
        self.set_filter_modal(1)
        self.set_filter_punc(1)
        self.set_filter_dirty(1)
        self.set_need_vad(1)
        self.set_voice_format(1)
        self.set_word_info(1)
        self.set_convert_num_mode(1)