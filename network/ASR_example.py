import os
import sys
import socket

sys.path.append(os.path.join(os.path.dirname(__file__), '../python'))

import flatbuffers
import VoiceCommand.ASRCommand
import VoiceCommand.VoiceOperationCommand
import VoiceCommand.UnionCommand


def VoiceCommandPacket(builder, name, type, command):
    # Serialize the FlatBuffer data. 
    name = builder.CreateString(name)
    VoiceCommand.ASRCommand.Start(builder)
    VoiceCommand.ASRCommand.AddName(builder, name)
    VoiceCommand.ASRCommand.AddCommand(builder, command)
    VoiceCommand.ASRCommand.AddCommandType(builder, type)
    return VoiceCommand.ASRCommand.End(builder)


def VoiceOperationCommandPacket(builder, object, time, location, action):
    object = builder.CreateString(object)
    time = builder.CreateString(time)
    location = builder.CreateString(location)
    action = builder.CreateString(action)
    VoiceCommand.VoiceOperationCommand.Start(builder)
    VoiceCommand.VoiceOperationCommand.AddObject(builder, object)
    VoiceCommand.VoiceOperationCommand.AddTime(builder, time)
    VoiceCommand.VoiceOperationCommand.AddLocation(builder, location)
    VoiceCommand.VoiceOperationCommand.AddAction(builder, action)
    return VoiceCommand.VoiceOperationCommand.End(builder)


def main():
    # create an AF_INET, STREAM socket (TCP)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # parameters
    ip = "127.0.0.1"
    port = 9001
    addr = (ip, port)
    buffer_size = 2048
    print('Socket Connected to IP: %s  port: %d' % (ip, port))
    # client_socket.connect(addr)
    builder = flatbuffers.Builder(0)
    operation = VoiceOperationCommandPacket(builder, "object", "time", "location", "action")
    voicecommand = VoiceCommandPacket(builder, "command",
                                      VoiceCommand.UnionCommand.UnionCommand().VoiceOperationCommand, operation)
    builder.Finish(voicecommand, b"asr2")
    # client_socket.send(builder.Output())


if __name__ == "__main__":
    main()
