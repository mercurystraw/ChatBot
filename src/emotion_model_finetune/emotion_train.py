import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import evaluate

# 在中文语料上训练的基于roberta架构的模型
model_name = 'hfl/chinese-roberta-wwm-ext' 
dataset_path = './dataset/OCEMOTION.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出7种基础情绪 开心、生气、难过、恐惧、惊讶、厌恶、喜欢
label_map = {
    "happiness": 0,
    "sadness": 1, 
    "anger": 2, 
    "fear": 3, 
    "surprise": 4, 
    "disgust": 5, 
    "like": 6
}
id2label = {v: k for k, v in label_map.items()}
label2id = label_map

num_labels = len(label_map)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
).to(device)

def tokenize_function(sample):
    return tokenizer(sample["text"], padding="max_length", truncation=True)

df = pd.read_csv(
    dataset_path, 
    sep='\t', 
    header=None, 
    names=['id', 'text', 'label_text'],
    on_bad_lines='skip'
)
#去除列中的空值
df = df[['text', 'label_text']].dropna()
# 转换成label
df['label'] = df['label_text'].map(label_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print(f"数据集加载成功，总计 {len(df)} 条有效样本。")
print("标签分布:")
print(df['label_text'].value_counts())

# 用Dataset
raw_dataset = Dataset.from_pandas(df)
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=['text', 'label_text'])

# 划分训练集和验证集
train_eval_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_eval_dataset["train"]
eval_dataset = train_eval_dataset["test"]

# 加载评估指标
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }

training_args = TrainingArguments(
    output_dir='./emotion_train_results',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32, # 根据您的GPU显存调整
    per_device_eval_batch_size=32,  # 根据您的GPU显存调整
    num_train_epochs=5, # 3个epoch
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1", # 使用 F1 分数来选择最佳模型
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\n--- 开始模型微调 ---")
trainer.train()
print("--- 微调完成！ ---")

save_path = "./emotion_model_ckpt"
trainer.save_model(save_path)
print(f"\n最佳模型已保存至: {save_path}")

# 使用 pipeline 进行简单的推理测试
from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification",
    model=save_path,
    tokenizer=save_path,
    device=0 if torch.cuda.is_available() else -1
)

test_text = "我考试考了第一名，今天可以爽吃了！"
prediction = emotion_classifier(test_text)
print(f"\n测试句子: '{test_text}'")
print(f"预测结果: {prediction}")