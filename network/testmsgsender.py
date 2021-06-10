import socket


class Server:
    def __init__(self, addr='127.0.0.1', port=8080):
        # AF_INET: IPV4
        # SOCK_STREAM: TCP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((addr, port))
        self.sock.listen(1)
        self.process_msg_func = self.defaultProcessMsg
        print('Initialize python')

    def start_to_listen(self):
        while True:
            conn, addr = self.sock.accept()
            print('Connect to %s. Start to receive message...' % str(addr))
            try:
                while True:
                    client_msg = conn.recv(1024).decode('utf-8')
                    returnMsg = self.process_msg_func(client_msg)
                    conn.send(returnMsg.encode('utf-8'))
                    if client_msg == 'shutdown':
                        break
                conn.close()
            except ConnectionError as conn_error:
                print('Disconnect to %s' % str(addr))
                conn.close()
            except UnicodeDecodeError as uni_error:
                print('Error Unicode not byte ? ')
                conn.close()

    def register_dealing_msg_func(self, process_msg_func):
        self.process_msg_func = process_msg_func


    def defaultProcessMsg(self, msg: str):
        return "Warning: not register process msg, using default process msg"


def customProcessMsg(msg):
    print("received msg", msg)
    return "process: " + msg + " over."


if __name__ == "__main__":
    server = Server(addr="127.0.0.1", port=9001)
    server.register_dealing_msg_func(customProcessMsg)
    server.start_to_listen()