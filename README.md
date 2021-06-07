# 语音转意图

## 整体架构
![图片](https://github.com/fkjslee/github_image/blob/main/pic4.jpg)

## 如何运行
环境：  
Windows==10  
Anaconda==4.10.1  
python==3.6
### 安装库
#### 安装pytorch:
查看自己的cuda版本, 命令: nvcc -V  
我的是cuda 11  
![图片](https://github.com/fkjslee/github_image/blob/main/pic5.png)  
选择自己的情况后, 在pytorch官网查看安装命令  
比如我的情况: Stable(1.8.1)+Windows10+pip...+CUDA11 (推荐使用pip安装 conda似乎有问题)  
![图片](https://github.com/fkjslee/github_image/blob/main/pic6.png)
#### 安装其他库
见根目录下的requirements.txt文件
可以一个一个单独安装, 但版本尽量不要差异太大, 某些库不同版本模型的位置不同  
更推荐使用命令(在根目录下)
```
pip3 install -r requirements.txt  
```
### 语音识别(asr)注册
需要去别的平台注册, 注册后的信息放在```"./asr/asrConfig.yml"```中
一个例子大概如下:
```
APPID: "xxx"
SECRET_ID: "xxx"
SECRET_KEY: "xxx"
ENGINE_MODEL_TYPE: "xxx"
SLICE_SIZE: 111
```
#### 不想要asr
可以直接运行根目录下的test_nlu.py, 输入命令后测试
