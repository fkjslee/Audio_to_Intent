import os
import sys
import socket

sys.path.append(os.path.join(os.path.dirname(__file__), '../python'))

import flatbuffers
from network.VoiceCommand import ASRCommand, VoiceOperationCommand, UnionCommand


class MsgSender:
    def __init__(self, addr=None, port=None):
        if addr is not None:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((addr, port))
        else:
            self.client_socket = None


    def send_msg(self, intent: str, entities: dict):
        if self.client_socket is None:
            return
        if intent == "move_object" and 'B-moved_object' in entities.keys() and 'B-moved_position' in entities.keys():
            builder = flatbuffers.Builder(0)
            operation = VoiceOperationCommandPacket(builder, entities['moved_object'], "time", "location", entities['moved_position'])
            voicecommand = VoiceCommandPacket(builder, "command",
                                              UnionCommand.UnionCommand().VoiceOperationCommand, operation)
            builder.Finish(voicecommand, b"asr2")
            self.client_socket.send(builder.Output())
            # self.client_socket.send(str(entities).encode("utf-8"))


    def __del__(self):
        if self.client_socket is not None:
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
    msgsender = MsgSender(addr="127.0.0.1", port=9001)
    msgsender.send_msg("move_object", {"moved_object": "fly1", "moved_position": "pos1"})
