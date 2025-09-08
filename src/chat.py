import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import os
from prompts import *
import pandas as pd
import re
import jieba  # 结巴中文分词

ChatGLM_model_path = "THUDM/chatglm2-6b"
emotion_model_path = "./emotion_model_finetune/emotion_model_ckpt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_history_len = 8 # 最大历史对话轮数
max_context_len = 2048 # 最大上下文长度，调整则对话可以更多轮

emotion_df = pd.read_excel("dataset\emotion_words.xlsx")
degree_df = pd.read_excel("dataset\degree_words.xlsx")

# 转换为字典
emotion_dict = dict(zip(emotion_df["词语"], emotion_df["强度"]))
degree_dict = dict(zip(degree_df["词语"], degree_df["等级"]))

def load_models():
    """
    加载 ChatGLM 和情感分析模型，需使用VPN或者设定镜像网站
    """      
    # 加载 ChatGLM 模型，使用 .half()或者quantize(4)量化，降低显存占用 
    print(f"正在从 '{ChatGLM_model_path}' 加载 ChatGLM-2-6B 模型...")
    chatglm_tokenizer = AutoTokenizer.from_pretrained(ChatGLM_model_path, trust_remote_code=True)
    chatglm_model = AutoModel.from_pretrained(ChatGLM_model_path, trust_remote_code=True).half()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行并行计算")
        chatglm_model = torch.nn.DataParallel(chatglm_model)
    chatglm_model = chatglm_model.to(device)

    # 设定模型为推理模式
    chatglm_model = chatglm_model.eval()

    # 使用 pipeline 加载微调的情感模型
    print(f"正在从 '{emotion_model_path}' 加载情感分析模型...")
    emotion_classifier = pipeline("text-classification", model=emotion_model_path, device=device)

    print("\n--- 所有模型加载完毕，对话系统准备就绪！ ---")
    
    return chatglm_tokenizer, chatglm_model, emotion_classifier

def get_emotion_intensity(user_input):
    total_intensity = 0

    previous_degree = 1  # 用来记录当前的程度词等级，默认为1（没有程度词）

    words = jieba.cut(user_input)  # 分词

    # 正则匹配情感词和程度词
    degree_pattern = "|".join(degree_dict.keys())  # 构建正则表达式：程度词
    emotion_pattern = "|".join(emotion_dict.keys())  # 构建正则表达式：情感词

    for word in words:
        # 匹配程度词
        if re.match(degree_pattern, word):
            previous_degree = degree_dict.get(word, 1)  # 更新当前的程度词等级
        # 匹配情感词
        elif re.match(emotion_pattern, word):
            intensity = emotion_dict.get(word, 1)
            total_intensity += intensity * previous_degree  # 情感词强度与当前程度词等级相乘
            previous_degree = 1  # 重置程度词等级

    return total_intensity



def get_emotion(user_input, emotion_classifier):
    # 情感分析
    emotion_results = emotion_classifier(user_input)
    # 获取得分最高的那个情感标签
    detected_emotion = emotion_results[0]['label']
    emotion_score = emotion_results[0]['score']
    print(f"检测到情感 -> {detected_emotion} (置信度: {emotion_score:.2f})")

    return detected_emotion, emotion_score



def start_chat():
    """此函数可以直接在终端测试对话系统"""
    chatglm_tokenizer, chatglm_model, emotion_classifier = load_models()
    # history是一个列表，每轮对话包含用户输入和AI回复（8轮的话就是长度16）
    history = []
    
    print("\n你好！我是结合了情感分析的 ChatGLM 助手。")
    print("输入 'quit' 或 'exit' 即可退出程序。")
    print("-" * 50)

    while True:
        user_input = input("【你】: ")
        if user_input.lower() in ["quit", "exit"]:
            print("感谢使用，再见！")
            break
            
        detected_emotion, emotion_score = get_emotion(user_input, emotion_classifier)
        intensity = get_emotion_intensity(user_input)

        print("detected_emotion: ", detected_emotion)
        print("emotion_score: ", emotion_score)
        print("intensity: ", intensity)
        
        if intensity < 5:
            strategy = "low"
        elif intensity < 10:
            strategy = "medium"
        else:
            strategy = "high"

        # 根据置信度构建策略性 Prompt，这里设定阈值为0.6，高于则表明比较确定情感，否则使用默认 Prompt
        strategy_instruction = emotion_strategy_prompt.get(detected_emotion).get(strategy) if emotion_score > 0.6 else emotion_strategy_prompt["default"]
        
        # 将指令和用户原始输入结合
        prompt_for_glm = f"({strategy_instruction})\n用户的输入是：'{user_input}'"
        print(f"输入ChatGLM的prompt: {prompt_for_glm}")

        if len(history) > max_history_len*2:
            history = history[-max_history_len*2:]
        # .chat 是ChatGLM特有对话接口
        response, history = chatglm_model.chat(
            chatglm_tokenizer,
            prompt_for_glm,
            history=history,
            max_length=max_context_len, 
            temperature=0.9,
        )
         
        print(f"【ChatGLM】: {response}\n")

if __name__ == "__main__":
    start_chat()
    