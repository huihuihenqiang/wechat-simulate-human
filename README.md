# wechat-simulate-human
This is a chatgpt-on-wechat based project to make wechat replies more human-like
## 更新2024.10 
使用多模式人工智能构建一个用于查询视频内容的交互式系统（MM-RAG）。创建一个复杂的问答系统，它可以处理、理解视频并与视频交互.流程如下：
从视频中提取帧和转录(情况一：视频里面有字幕VTT)，使用Whisper模型生成转录（情况二：视频里面没有字幕但是有声音），并使用大视觉语言模型(LVLMs)创建字幕（视频里面只有画面）。
使用BridgeTower进行多模态嵌入：为图像标题对创建联合嵌入，生成512维度的向量，测量相似性（cosin），并可视化高维嵌入(UMAP)
构建多模态向量数据库:使用LanceDB和LangChain实现多模态检索，对多模态数据进行相似性搜索，将构建好的数据输入到lancedb中，可实现快捷查找。
利用大型视觉语言模型（lvlm）：LLaVA，并实现图像字幕、视觉问题回答和多回合对话，将上一个步骤中查询出来的(图像+问题+字幕)都当作输入给llava,回答问题。
![图片](https://github.com/user-attachments/assets/b240ab27-6729-46a1-9909-e7a5e55ab809)
```python
from gradio_utils import get_demo
#You will need to restart the kernel each time you rerun this cell;
#otherwise, the port will not be available.

debug = False # change this to True if you want to debug

demo = get_demo()
demo.launch(server_name="0.0.0.0", server_port=9999, debug=debug)
```
** 效果 **：
![图片](https://github.com/user-attachments/assets/aa9bf5a0-fdde-414a-9d41-575c5b01882d)
注：这个会后期加入到聊天回复项目中。


## 更新2024.10 🤖 Chatbot Fine-Tuning with LoRA
## 📊 数据准备

首先，我们需要准备聊天记录数据。数据以**问答对**的形式组织，如下所示：

```json
[
  {"question": "你的名字是什么？", "answer": "我叫小助手。"},
  {"question": "今天天气怎么样？", "answer": "今天阳光明媚。"}
]
```
在训练之前，我们需要将原始数据进行 Tokenization（将文本转换为模型可理解的数字形式）。GPT 模型需要将每个句子分解为词汇或词片段，最终生成对应的 token ids 作为输入。
```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_function(data):
    return tokenizer(data["question"], data["answer"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function)
```
接下来是 LoRA 微调。LoRA 可以让我们在不改变整个 GPT 模型的情况下，只针对特定层进行微调，从而节省计算资源。训练过程包括以下步骤：
设置模型参数：加载预训练的 GPT (GPT-2)模型并应用 LoRA。
配置训练参数：指定训练步数、学习率、批次大小等超参数。
```python
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

model = GPT2LMHeadModel.from_pretrained("gpt2")
config = LoraConfig(...)
model = get_peft_model(model, config)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

``` 
❗ 遇到的问题

    Attention Mask 和 Pad Token：
        我在训练时遇到了警告：attention mask and the pad token id were not set。为了解决这个问题，我确保为输入设置了正确的 attention mask。

    标签问题：
        在微调过程中，最初我对是否需要标签感到困惑。通过查阅文档，我了解到在某些情况下，不需要明确的标签。对于聊天记录数据来说，问答对中的 "answer" 可以被视为隐式标签。
        或者可以将目的作为标签。

    模型输出：
        在训练后，模型的输出有时会出现重复或不连贯的句子。通过调整模型参数和训练数据，可以进一步改善输出质量。
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

- 然后将本地的\channel\wechat\wechat_channel.py换成我们这个项目的wechat_channel.py

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


