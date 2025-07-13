import os
from dotenv import load_dotenv, find_dotenv
from zhipuai import ZhipuAI

# 加载 .env 文件中的环境变量
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print(".env 文件未找到，请确保 .env 文件与脚本在同一目录下")

# 检查是否正确加载了 API Key
api_key = os.environ.get("ZHIPUAI_API_KEY")

# 初始化 ZhipuAI 客户端
client = ZhipuAI(
    api_key=api_key  # 使用加载的 API Key
)

def gen_glm_params(prompt):
    messages = [{"role": "user", "content": prompt}]
    return messages

def get_completion(prompt, model="glm-4-plus", temperature=0.95):
    messages = gen_glm_params(prompt)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        if len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "generate answer error"
    except Exception as e:
        print("调用失败:", e)
        return "调用失败"

# 交互式问答
while True:
    prompt = input("用户:   ")
    if prompt.lower() == "exit":
        break
    response = get_completion(prompt)
    print("助手:  ", response)




    
