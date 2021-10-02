# 语音转意图

## 整体架构
![图片](https://github.com/fkjslee/github_image/blob/main/pic4.jpg)
关于NLU部分结构图解
数据流程如图
![图片](https://github.com/fkjslee/github_image/blob/main/architecture.png)
模型整个图中括号内的数字表示tensor的shape

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

## 命令行参数
- **--do_load**  
    load之前训练的模型，一般存在根目录下*_model文件中, 默认False
- **--do_valid**  
  是否用数据集下的valid数据集验证结果，一般可以用来预判一下模型是否正常(acc>0.9一般就没啥问题), 默认False
- **--device**  
  模型是在cpu或者哪块gpu上运行，比如 **--device cuda:1** 则是在1号GPU运行, 默认"cuda 0"

## Replay
把需要重放的文件放在"./replay"文件夹下，运行replay.py，命令行参数的普通运行的命令行参数一样

## 结果说明
```
asr result: 语音转文字的识别结果
jieba cut: 使用jieba分词后的结果
predict intent: 预测的意图
predict slot: 预测的实体，和分词的顺序对应，表示每个词属于哪一类
```
