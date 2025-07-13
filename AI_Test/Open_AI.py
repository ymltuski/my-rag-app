import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv(find_dotenv())

# 初始化 OpenAI 客户端（使用小爱代理接口）
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # 从环境变量读取 API Key
    base_url="https://xiaoai.plus/v1",         # 设置为小爱提供的接口地址
)

# 初始化消息列表
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# 交互式对话
while True:
    user_input = input("用户:   ")
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        response = completion.choices[0].message.content
        print("助手:  ", response)

        messages.append({"role": "assistant", "content": response})

    except Exception as e:
        print("调用失败:", e)
