import os
import sys
import socket
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../python'))

import flatbuffers
import logging
from network.VoiceCommand import ASRCommand, VoiceOperationCommand, UnionCommand

logger = logging.getLogger(__name__)


class MsgSender:
    def __init__(self, addr=None, port=None):
        self.addr = addr
        self.port = port

    def send_msg(self, msg: str):
        if self.addr is not None:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.addr, self.port))
        else:
            return
        for _ in range(1):
            try:
                logger.warning("Start to send message, message = {}".format(msg))
                builder = flatbuffers.Builder(0)
                operation = VoiceOperationCommandPacket(builder, msg, "time", "location", 'position')
                voicecommand = VoiceCommandPacket(builder, "command",
                                                  UnionCommand.UnionCommand().VoiceOperationCommand, operation)
                builder.Finish(voicecommand, b"asr2")
                self.client_socket.send(builder.Output())
                buffer = self.client_socket.recv(1024)
                buffer = buffer.decode("utf-8")
                return buffer
            except Exception as e:
                logger.warning(e)
                logger.warning("Send message failed, message = {}".format(msg))
                time.sleep(0.25)


    def __del__(self):
        if hasattr(self, "client_socket") and self.client_socket is not None:
            self.client_socket.close()


def VoiceCommandPacket(builder, name, type, command):
    # Serialize the FlatBuffer data.
    name = builder.CreateString(name)
    ASRCommand.Start(builder)
    ASRCommand.AddName(builder, name)
    ASRCommand.AddCommand(builder, command)
    ASRCommand.AddCommandType(builder, type)
    return ASRCommand.End(builder)


def VoiceOperationCommandPacket(builder, object, time, location, action):
    object = builder.CreateString(object)
    time = builder.CreateString(time)
    location = builder.CreateString(location)
    action = builder.CreateString(action)
    VoiceOperationCommand.Start(builder)
    VoiceOperationCommand.AddObject(builder, object)
    VoiceOperationCommand.AddTime(builder, time)
    VoiceOperationCommand.AddLocation(builder, location)
    VoiceOperationCommand.AddAction(builder, action)
    return VoiceOperationCommand.End(builder)


if __name__ == "__main__":
    msgsender = MsgSender(addr="127.0.0.1", port=9010)
    print("start to send")
    msgsender.send_msg("输 入 不 太 好")
    # import json
    # msgsender.send_msg(json.dumps({"sentence1": "你 好", "sentence2": "早 上 好"}))
