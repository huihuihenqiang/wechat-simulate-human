# import pandas as pd
# from datasets import load_dataset
#
# # 1. 加载数据集
# dataset = load_dataset('swulling/gsm8k_chinese')
#
# # 3. 转换为Pandas DataFrame
# train_df = dataset['train'].to_pandas()
# val_df = dataset['test'].to_pandas()
#
# # 4. 保存为CSV（确保UTF-8编码）
# train_df.to_csv('train.csv', index=False, encoding='utf-8-sig')
# val_df.to_csv('validation.csv', index=False, encoding='utf-8-sig')
#
# print("数据集已成功保存为 train_dataset.csv 和 validation_dataset.csv")
# from datasets import load_dataset
#
# # 修改为你的实际路径
# file_path = 'train.csv'  # 替换为你的实际路径
#
# # 加载本地 CSV 文件
# ds = load_dataset('csv', data_files={'train': file_path})
# SYSTEM_PROMPT = """
# 按照如下格式生成：
# <think>
# ...
# </think>
# <answer>
# ...
# </answer>
# """
# def process_data(data):
#     data = data.map(lambda x: {
#         'prompt': [
#             {'role': 'system', 'content': SYSTEM_PROMPT},
#             {'role': 'user', 'content': x['question_zh-cn']}
#         ],
#         'answer': x['answer_only']
#     })
#     return data
# # 如果 process_data 需要 Dataset 对象可以直接传递
# data = process_data(ds['train'])
#
# print(len(data))
# print(data[0])

# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# # 加载模型和分词器
# model_name = "Qwen2.5-0.5B-Instruct"  # 如 "gpt2", "deepseek" 等
# model = AutoModelForCausalLM.from_pretrained("output/checkpoints-500")
# tokenizer = AutoTokenizer.from_pretrained("output/checkpoints-500")
#
# # 输入文本
# input_text = "小明有5个苹果，吃了1个，还剩下几个？"
# inputs = tokenizer(input_text, return_tensors="pt")
#
# # 生成输出
# outputs = model.generate(**inputs, max_new_tokens=50)
# print(tokenizer.decode(outputs[0]))

from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载微调后的检查点
checkpoint_path = "/root/autodl-tmp/output/checkpoint-500"  # 绝对路径
model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# 输入与生成
input_text = "小明有5个苹果，吃了1个，还剩下几个？"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))