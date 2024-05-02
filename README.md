# wechat-simulate-human
This is a chatgpt-on-wechat based project to make wechat replies more human-like

## 中文介绍
## 🎨 项目背景与简介


在这个开源的世界里，我们都是站在巨人的肩膀上。`wechat-simulate-human` 是对[[chatgpt-on-wechat](https://github.com/zhayujie/chatgpt-on-wechat)]的一次深情致敬与革新之旅。我在此基础上，使得微信回复的消息更加拟人化，有具体三个小细节，让这个工具箱更加闪耀！主要的灵感来源于[[issues](https://github.com/zhayujie/chatgpt-on-wechat/issues/1078)]和@[[Namarimizu](https://github.com/Namarimizu)]的所有参与者和作者,本项目可以实现的功能都是在本地部署的，其他部署的方式还请自行适配。


## ✨ 新增特性


- 🌈 **模拟人类回复微信**: 人类回复微信不会一大段回复，而是会一句一句的回复。

- 🔧 **延长回复时间和回复表情以及表情包**: 人类的聊天是多使用表情和表情包的，这样更逼真。根据句子的长度决定回复时间。

- 📊 **使用聊天记录训练情感机器人方法**: 使用另外一个开源项目[[WeChatMsg](https://github.com/LC044/WeChatMsg)]获取聊天记录数据训练模型，这里使用LinkAI。


## 🛠️ 如何使用

- 克隆原始仓库：`git clone https://github.com/zhayujie/chatgpt-on-wechat.git`，并且根据大佬的仓库的指导部署项目，这里推荐使用本地部署，大模型使用LinkAI。
- 首先保证你能正常在本地正常运行[[chatgpt-on-wechat](https://github.com/zhayujie/chatgpt-on-wechat)]，如下图，能正常输出一大段文字。

![图片](https://github.com/huihuihenqiang/wechat-simulate-human/assets/99072450/1b91d75e-fafe-4b07-82de-1174e49d7e97)

- 然后将本地的\channel\wechat_channel.py换成我们这个项目的wechat_channel.py

- 然后在本地的根目录下添加我们这个项目的.\resources\stickers\，，就完成了。如图：

![图片](https://github.com/huihuihenqiang/wechat-simulate-human/assets/99072450/04fcbd0b-04bc-4d47-ac0c-0bf530d155bd)

- 将自己的propmt增加一段话：

`说话要带上微信表情或者颜文字，也不是每一句都带上，有时候带上，有时候不用带，带表情的情况更多。你的回话格式是  ：语言+ %情感%，比如「我很开心 %愉快%」。只能用这10种情感，不要同义词替换，就是这10个词语，如果有其他情感就直接回复，不用这种格式。语言就是你要说的话，不用处理。%情感%就是你对话时的情感，放在%之间，只有以下10种：「%无语%、%惊讶%、%难过%、%愉快%、%疑惑%、%安慰%、%害羞%、%担忧%、%爱你%、%关心%」，若都不符合则不说出%情感%。如果你要回复语音的时候，不说出%情感%。必须用这种格式回话。你只能回复这10种「%无语%、%惊讶%、%难过%、%愉快%、%疑惑%、%安慰%、%害羞%、%担忧%、%爱你%、%关心%」，若都不符合则不说出%情感%，不用这种格式，一定要记住。你可以在自己的回复的话上带上表情。说语言的时候不要加上“「」”，就干干净净的语言就行。只能用这10种情感，不要同义词替换，就是这10个词语，如果有其他情感就直接回复，不用这种格式。表情的生成在逗号的前后。`

- 如果你使用了LinkAI,你还可以将自己的聊天记录做成训练样本导入进去(可选)：

  首先下载WeChatMsg的软件，然后将数据制作成


- 最终的结果展示：

![图片](https://github.com/huihuihenqiang/wechat-simulate-human/assets/99072450/daf21818-2404-4c34-9341-43aeff68a14d)



## 🤝 贡献指南


我们非常欢迎任何形式的贡献！无论是提交bug报告、提出新特性建议，还是直接提交代码，您的每一份努力都将使这个项目更加完善。最好是能将这个做成一个插件，我的能力和精力不够，欢迎大家参与。


## 🎭 致谢与彩蛋


最后，别忘了探索我们的隐藏彩蛋——一个精心设计的复活节彩蛋，等待着有缘人在源码深处发现它。找到后，记得在Issues中分享你的喜悦哦！


---


## 🇺🇸 English Introduction


Dive into the exciting enhancements of `Enhanced Feature Factory`, built upon the shoulders of the [original project link], refining its core strengths and introducing game-changing features that elevate the open-source experience.


[... Continue with the English version of the content ...]
