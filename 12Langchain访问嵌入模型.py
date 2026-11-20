from langchain_ollama import OllamaEmbeddings

# 嵌入模型只能用 bge-m3：qwen3:4b 是聊天模型，不具备 embedding 能力，会报 501
model = OllamaEmbeddings(model="bge-m3")

print(model.embed_query("我喜欢你"))
print(model.embed_documents(["我喜欢你",'我稀饭你','晚上吃啥']))