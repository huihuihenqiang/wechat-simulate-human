import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from sklearn.metrics import accuracy_score
from transformers import DataCollatorForLanguageModeling
from transformers import AutoProcessor
import gc
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# 1. 数据加载与预处理
def parse_conversations(text):
    import ast
    # 先处理换行符（替换为逗号）并移除可能存在的多余空格
    processed_text = text.replace('\n', ', ').replace('} {', '}, {')
    try:
        return ast.literal_eval(processed_text)
    except SyntaxError as e:
        print(f"解析失败的文本: {processed_text}")  # 调试输出
        raise e


# 读取CSV文件
train_df = pd.read_csv('train.csv', converters={'conversations': parse_conversations})
val_df = pd.read_csv('validation.csv', converters={'conversations': parse_conversations})

# 转换为HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})
print('数据加载完成')
del train_df, val_df
gc.collect()
torch.cuda.empty_cache()
# 2. 加载模型和分词器
model_path = "Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 保持混合精度
    device_map="auto",  # 自动设备分配
    low_cpu_mem_usage=True,  # 优化CPU内存
    trust_remote_code=True
)
print("模型加载没问题")

# 3. LoRA配置 <button class="citation-flag" data-index="2"><button class="citation-flag" data-index="5"><button class="citation-flag" data-index="9">
peft_config = LoraConfig(
    r=4,  # 进一步降低到4
    lora_alpha=8,  # 保持比例
    target_modules=["q_proj", "v_proj"],  # 仅保留最关键模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
#model = get_peft_model(model, peft_config)
print("模型加载没问题2")
# 在模型加载后释放缓存
# 2. 启用梯度检查点
model.gradient_checkpointing_enable()
model = get_peft_model(model, peft_config)
torch.cuda.empty_cache()
# 4. 数据预处理函数 <button class="citation-flag" data-index="4"><button class="citation-flag" data-index="7">
def preprocess_function(examples):
    inputs = []
    for conv in examples['conversations']:
        # 构建对话历史
        history = []
        for turn in conv:
            if turn['role'] == 'system':
                history.append(f"<|system|>\n{turn['content']}")
            elif turn['role'] == 'user':
                history.append(f"<|user|>\n{turn['content']}")
            else:
                history.append(f"<|assistant|>\n{turn['content']}")
        inputs.append('\n'.join(history))

    # 使用Qwen的聊天模板 <button class="citation-flag" data-index="4">
    return tokenizer(
        inputs,
        padding="max_length",  # 改为动态填充
        truncation=True,
        max_length=256,  # 最大长度减半
        return_tensors="pt"
    )

print('数据处理完成')
# 应用预处理
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['conversations']
)


# 5. 定义评估指标 <button class="citation-flag" data-index="5">
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 转换为float32
    predictions = torch.from_numpy(predictions).float()
    labels = torch.from_numpy(labels).long()  # 标签应为长整型
    loss = torch.nn.functional.cross_entropy(
        predictions.view(-1, predictions.size(-1)),  # 调整形状为 [batch_size*seq_len, vocab_size]
        labels.view(-1)  # 调整形状为 [batch_size*seq_len]
    )
    torch.cuda.empty_cache()
    return {"eval_loss": loss.item()}


# 6. 配置训练参数 <button class="citation-flag" data-index="6"><button class="citation-flag" data-index="7">
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    output_dir="./qwen2.5_lora_finetune",
    per_device_eval_batch_size=1,
    eval_accumulation_steps=10,  # 分批次评估，避免一次性加载全部数据
    num_train_epochs=3,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=10,
    eval_strategy="epoch",  # 完全关闭评估避免显存峰值
    eval_steps=None,
    save_strategy="epoch",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16_full_eval=True,  # 确保评估时也使用混合精度
    fp16=True,
    push_to_hub=False,
    deepspeed="ds_config.json",  # 指定配置文件路径
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# 7. 创建SFTTrainer <button class="citation-flag" data-index="5"><button class="citation-flag" data-index="7">
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=processor,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# 8. 开始训练
trainer.train()

# 9. 保存最终模型
trainer.save_model("./qwen2.5_lora_finetuned")