# network

### 使用方法
#### 1. 需要将识别结果发送到服务器
```
python main.py --command_server_addr "127.0.0.1"  --command_server_port 9001
```

#### 2. 不需要将识别结果发送到服务器
```
python main.py
```

#### 3. 测试
① 运行服务器
```
python network/testmsgsender.py  
```
![图片](https://github.com/fkjslee/github_image/blob/main/pic7.png)  
由于测试的时候发送的是flatbuffers的结果, 会乱码，若仅为了测试结果是否正确，把network下的msgsender.MsgSender.sendMsg中发送flatbuffers的方式改成普通的发送方式(即被注释掉的方法)即可。或者在接收端自己解码flatbuffers。

② 运行意图分类程序
```
python main.py --command_server_addr "127.0.0.1"  --command_server_port 9001
```

![图片](https://github.com/fkjslee/github_image/blob/main/pic8.png)

### 结果说明
见 [根目录README下的结果说明](https://github.com/fkjslee/Audio_to_Intent#%E4%B8%8D%E6%83%B3%E8%A6%81asr)