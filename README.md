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

  首先下载WeChatMsg的软件，然后将数据制作成template.csv的形式导入到LINkAi中的知识库中，这可能会使得你得机器人更像你本人，包括你的语言习惯什么的。
  ![图片](https://github.com/huihuihenqiang/wechat-simulate-human/assets/99072450/00a3427f-a530-4b08-b742-0cd39aaff6ab)

- **最终的结果展示**：

![图片](https://github.com/huihuihenqiang/wechat-simulate-human/assets/99072450/daf21818-2404-4c34-9341-43aeff68a14d)



## 🤝 贡献指南


我们非常欢迎任何形式的贡献！无论是提交bug报告、提出新特性建议，还是直接提交代码，您的每一份努力都将使这个项目更加完善。最好是能将这个做成一个插件，我的能力和精力不够，欢迎大家参与。


## 🎭 致谢与彩蛋


最后，别忘了探索我们的隐藏彩蛋，等待着有缘人在源码深处发现它。找到后，记得在Issues中分享你的喜悦哦！


---


## 🇺🇸 English Introduction
## 🎨 Project Background and Overview

In this open-source world where we all stand on the shoulders of giants, `wechat-simulate-human` pays a heartfelt tribute to and embarks on an innovative journey inspired by [[chatgpt-on-wechat](https://github.com/zhayujie/chatgpt-on-wechat)]. Building upon its foundation, I've enhanced the WeChat responses to be more human-like with three subtle improvements, making this toolbox shine even brighter! The primary inspiration stems from [[issues](https://github.com/zhayujie/chatgpt-on-wechat/issues/1078)] and the contributions from all participants and the author, @[[Namarimizu](https://github.com/Namarimizu)]. This project's functionalities are implemented for local deployment; adaptations for other deployment methods are left to individual users.

## ✨ New Features

- 🌈 **Simulated Human-like Replies on WeChat**: Unlike automated bulk replies, messages are sent sentence by sentence, mimicking human behavior.

- 🔧 **Extended Response Times, Emojis, and Stickers**: To enhance realism, responses include emojis and stickers, with reply times adjusted based on message length.

- 📊 **Training Method for Emotional Chatbots using Chat Logs**: Leveraging another open-source project, [[WeChatMsg](https://github.com/LC044/WeChatMsg)], to gather chat data for model training with LinkAI.

## 🛠️ How to Use

1. Clone the original repository: `git clone https://github.com/zhayujie/chatgpt-on-wechat.git`, then follow the guide from the original repository to set up the project, preferably locally with LinkAI for large models.
2. Ensure you can run [[chatgpt-on-wechat](https://github.com/zhayujie/chatgpt-on-wechat)] locally, as depicted below, successfully generating lengthy text responses.

   ![Image](https://github.com/huihuihenqiang/wechat-simulate-human/assets/99072450/1b91d75e-fafe-4b07-82de-1174e49d7e97)

3. Replace your local `\channel\wechat_channel.py` with the `wechat_channel.py` from this project.
4. Add our project's `\resources\stickers\` directory to your local root folder, as shown:

   ![Image](https://github.com/huihuihenqiang/wechat-simulate-human/assets/99072450/04fcbd0b-04bc-4d47-ac0c-0bf530d155bd)

5. Append to your prompt:

   Speak with WeChat emojis or emoticons, not every sentence, sometimes included, sometimes not, with emojis being more frequent. Your response format is `Language + %Emotion%,` e.g., "I'm happy %joyful%". Stick to these ten emotions; avoid synonyms; use these exact ten words. If other emotions arise, respond directly without the format. The language is what you want to say, unaltered. `%Emotion%` denotes the emotion during conversation, enclosed in %, limited to these ten: `%Speechless%, %Surprised%, %Sad%, %Joyful%, %Puzzled%, %Comforting%, %Shy%, %Worried%, %Love%, %Care%`. If none fit, omit %Emotion%. For voice replies, exclude %Emotion%. Adhere strictly to this format. Only use these ten `%Emotions%;` if none apply, disregard the format. Remember, you can embellish your responses with emojis. Exclude "「」" when speaking; keep it clean. Limit to these ten emotions; no synonyms; if other emotions arise, respond directly. Emojis are generated before and after commas.

6. Optionally, if using LinkAI, you can turn your chat history into training samples:

   Download WeChatMsg, convert data into a `template.csv` format, and import it into LINkAI's knowledge base. This may personalize your bot to mirror your language habits.
   
   ![Image](https://github.com/huihuihenqiang/wechat-simulate-human/assets/99072450/00a3427f-a530-4b08-b742-0cd39aaff6ab)

7. **Final Showcase**:

   ![Image](https://github.com/huihuihenqiang/wechat-simulate-human/assets/99072450/daf21818-2404-4c34-9341-43aeff68a14d)

## 🤝 Contribution Guide

We warmly welcome contributions of any kind! Whether it's submitting bug reports, suggesting new features, or directly contributing code, your efforts will help refine this project. Ideally, transforming this into a plugin would be great, but due to limitations in my capacity and energy, I invite everyone to participate.

## 🎭 Acknowledgments and Easter Eggs

Lastly, don't forget to explore our hidden easter eggs, awaiting discovery by the discerning in the depths of our source code. Share your excitement in the Issues section once found!

---


