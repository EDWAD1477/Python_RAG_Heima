# -*- coding: utf-8 -*-
"""
08 LangChain 调用阿里云通义千问
=============================================================

【原来的代码为什么报错？—— 大白话版】

原写法：
    from langchain_community.llms.tongyi import Tongyi
    model = Tongyi(model="qwen3.8-flash")
    model.invoke("你是谁")

报错：
    400 InvalidParameter: url error, please check url！

问题不在密钥、不在额度，而在"通道对不上"：

    阿里云的通义模型有两个"门"：

    门 1：原生通道（阿里自己的专用接口）
        https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation
        - Tongyi 这个类只会敲这扇门
        - 只认自家列表里的模型：qwen-max / qwen-plus / qwen-turbo ...

    门 2：OpenAI 兼容通道（阿里开的"通用窗口"）
        https://dashscope.aliyuncs.com/compatible-mode/v1
        - 把自家模型伪装成 OpenAI 的模型，谁会用 OpenAI 接口都能来调
        - qwen3.8-flash 只在"门 2"提供服务

    你让 Tongyi（只会敲门 1）去调 qwen3.8-flash（只在门 2 服务），
    服务端找不到对应模型，就丢了个含糊的 "url error" ——
    字面像"网址写错了"，真实含义是"这个模型不走这条通道"。

【解决办法】
把敲门方式换掉：用 ChatOpenAI（它天生就会说 OpenAI 那套接口语言），
把 base_url 指向"门 2"的地址。因为阿里这扇门长得跟 OpenAI 一模一样，
所以原本调 OpenAI 的类，照样能调通通义。

【密钥从哪来】
    1) 系统环境变量 DASHSCOPE_API_KEY（已配置，优先用）
    2) 项目根目录的 .env 文件（用 load_dotenv() 读进来兜底）
    注意：load_dotenv() 默认"不覆盖"已有环境变量，
          所以系统里新配的 key 会赢，.env 里的旧 key 不会顶掉它。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 第 1 步：把 .env 里的变量读进当前进程（没有 .env 也不会报错，只是返回 False）
load_dotenv()

# 第 2 步：取密钥。系统环境变量里有就用它，没有才用 .env 里的
api_key = os.getenv("DASHSCOPE_API_KEY")


# 第 3 步：创建模型对象
#   - model   ：模型名，写阿里云控制台里能看到的那个
#   - api_key ：密钥
#   - base_url：关键！指向"OpenAI 兼容通道"，不是原生通道
model = ChatOpenAI(
    model="qwen3.8-flash",
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 第 4 步：调用模型
# ChatOpenAI 返回的是 AIMessage 对象，正文在 .content 里（不是直接返回字符串）
res = model.invoke("你是谁")
print(res.content)


# ============================================================
# 附：原始代码（保留下来方便对照，想跑它需要把模型换成 qwen-max，
#     但 qwen-max 的免费额度已耗尽，会报 403 FreeTierOnly）
# ============================================================
# from langchain_community.llms.tongyi import Tongyi
#
# model = Tongyi(model="qwen3.8-flash")   # ← 这行就是 url error 的根源
# res = model.invoke(input="你是谁")
# print(res)
#
# 若一定要用 Tongyi，只能配原生通道支持的模型，例如：
#     model = Tongyi(model="qwen-max")
# 但它走的是"门 1"，而门 1 的免费额度已用完（403 AllocationQuota.FreeTierOnly）。
