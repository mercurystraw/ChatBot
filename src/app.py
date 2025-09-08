import gradio as gr
import uuid
from datetime import datetime
from chat import *
from prompts import emotion_strategy_prompt

# Linux   export HF_ENDPOINT=https://hf-mirror.com
# Windows $env:HF_ENDPOINT = "https://hf-mirror.com"
# 全局变量存储模型和对话历史
chatglm_tokenizer = None
chatglm_model = None
emotion_classifier = None

sessions = {}  # 存储所有对话会话
current_session_id = None

def initialize_models():
    """初始化模型"""
    global chatglm_tokenizer, chatglm_model, emotion_classifier
    if chatglm_tokenizer is None:
        chatglm_tokenizer, chatglm_model, emotion_classifier = load_models()

def create_new_session():
    """创建新的对话会话"""
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    sessions[session_id] = {
        'title': f"新对话 {timestamp}",
        'history': [],  # ChatGLM的历史记录格式
        'messages': []  # 用于显示的消息列表
    }
    return session_id

def get_session_list():
    """获取会话列表"""
    return [(v['title'], k) for k, v in sessions.items()]

def chat_response(message, session_id):
    """处理对话响应"""
    if not message.strip():
        return "", sessions[session_id]['messages']
    
    # 情感分析
    detected_emotion, emotion_score = get_emotion(message, emotion_classifier)
    intensity = get_emotion_intensity(message)
    
    # 确定策略
    if intensity < 5:
        strategy = "low"
    elif intensity < 10:
        strategy = "medium"
    else:
        strategy = "high"
    
    # 构建策略性 Prompt
    strategy_instruction = emotion_strategy_prompt.get(detected_emotion, {}).get(strategy, emotion_strategy_prompt["default"]) if emotion_score > 0.6 else emotion_strategy_prompt["default"]
    prompt_for_glm = f"({strategy_instruction})\n用户的输入是：'{message}'"
    
    # 获取当前会话的历史记录
    history = sessions[session_id]['history']
    
    # 限制历史记录长度
    if len(history) > max_history_len * 2:
        history = history[-max_history_len * 2:]
    
    # 调用ChatGLM生成响应
    response, updated_history = chatglm_model.chat(
        chatglm_tokenizer,
        prompt_for_glm,
        history=history,
        max_length=max_context_len,
        temperature=0.9,
    )
    
    # 更新会话历史
    sessions[session_id]['history'] = updated_history
    sessions[session_id]['messages'].append({"role": "user", "content": message})
    sessions[session_id]['messages'].append({"role": "assistant", "content": response})
    
    # 更新会话标题（使用第一条用户消息的前20个字符）
    if len(sessions[session_id]['messages']) == 2:
        sessions[session_id]['title'] = message[:20] + "..." if len(message) > 20 else message
    
    return "", sessions[session_id]['messages']

def load_session(session_id):
    """加载指定会话"""
    global current_session_id
    if session_id in sessions:
        current_session_id = session_id
        return sessions[session_id]['messages']
    return []

def format_messages_for_chatbot(messages):
    """将消息格式转换为Gradio Chatbot组件所需的格式"""
    chatbot_messages = []
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            user_msg = messages[i]['content']
            assistant_msg = messages[i + 1]['content']
            chatbot_messages.append([user_msg, assistant_msg])
    return chatbot_messages

# 自定义CSS样式
custom_css = """
#session-list {
    max-height: 500px;
    overflow-y: auto;
}

.session-item {
    padding: 8px 12px;
    margin: 2px 0;
    border-radius: 6px;
    cursor: pointer;
    transition: background-color 0.2s;
}

.session-item:hover {
    background-color: rgba(0, 0, 0, 0.05);
}

.session-item.selected {
    background-color: rgba(0, 123, 255, 0.1);
    border-left: 3px solid #007bff;
}

#chatbot {
    height: 600px;
}

#msg-input {
    border-radius: 20px;
}

.main-container {
    display: flex;
    height: 100vh;
}

.sidebar {
    width: 300px;
    background-color: #f8f9fa;
    padding: 20px;
    border-right: 1px solid #e9ecef;
}

.chat-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 20px;
}
"""

def create_interface():
    """创建Gradio界面"""
    # 创建默认会话
    default_session_id = create_new_session()
    global current_session_id
    current_session_id = default_session_id
    
    with gr.Blocks(css=custom_css, title="ChatGLM 情感对话助手") as interface:
        gr.Markdown("# ChatGLM 情感对话助手")
        
        # 状态变量
        session_state = gr.State(default_session_id)
        
        with gr.Row(equal_height=True):
            # 左侧边栏
            with gr.Column(scale=1, elem_id="sidebar"):
                gr.Markdown("### 对话历史")
                
                new_chat_btn = gr.Button("+ 新建对话", variant="primary")
                
                # 会话列表
                session_dropdown = gr.Dropdown(
                    choices=get_session_list(),
                    value=default_session_id,
                    label="选择对话",
                    elem_id="session-list",
                    interactive=True,
                    allow_custom_value=True
                )
            
            # 右侧对话区域
            with gr.Column(scale=3, elem_id="chat-area"):
                # 对话框
                chatbot = gr.Chatbot(
                    value=[],
                    elem_id="chatbot",
                    height=600,
                    bubble_full_width=False,
                    show_copy_button=True
                )
                
                # 输入区域
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入你的消息...",
                        elem_id="msg-input",
                        scale=4,
                        lines=1,
                        max_lines=5
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
        
        # 事件处理
        def handle_new_chat():
            new_session_id = create_new_session()
            return (
                new_session_id,  # 更新session_state
                get_session_list(),  # 更新dropdown choices
                new_session_id,  # 更新dropdown value
                []  # 清空chatbot
            )
        
        def handle_session_change(session_id):
            messages = load_session(session_id)
            chatbot_messages = format_messages_for_chatbot(messages)
            return session_id, chatbot_messages
        
        def handle_message(message, session_id):
            _, updated_messages = chat_response(message, session_id)
            chatbot_messages = format_messages_for_chatbot(updated_messages)
            # 更新会话列表（因为标题可能会变）
            updated_choices = get_session_list()
            return "", chatbot_messages, updated_choices
        
        # 绑定事件
        new_chat_btn.click(
            handle_new_chat,
            outputs=[session_state, session_dropdown, session_dropdown, chatbot]
        )
        
        session_dropdown.change(
            handle_session_change,
            inputs=[session_dropdown],
            outputs=[session_state, chatbot]
        )
        
        # 发送消息事件
        def send_message(message, session_id):
            if message.strip():
                return handle_message(message, session_id)
            return message, format_messages_for_chatbot(sessions[session_id]['messages']), get_session_list()
        
        send_btn.click(
            send_message,
            inputs=[msg_input, session_state],
            outputs=[msg_input, chatbot, session_dropdown]
        )
        
        msg_input.submit(
            send_message,
            inputs=[msg_input, session_state],
            outputs=[msg_input, chatbot, session_dropdown]
        )
    
    return interface

if __name__ == "__main__":
    # 预加载模型（可选，也可以在第一次对话时加载）
    print("正在启动 ChatGLM 情感对话助手...")
    print("首次使用时需要加载模型，可能需要一些时间...")

    initialize_models()
    
    interface = create_interface()
    interface.launch(
        # server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,
        share=False,  # 设置为True可以生成公网链接
        inbrowser=True  # 自动打开浏览器
    )