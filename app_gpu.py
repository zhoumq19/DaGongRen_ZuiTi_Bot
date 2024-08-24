# 向量模型下载
import os
from modelscope import snapshot_download
# 导入所需的库
from typing import List
import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


# 模型下载功能,如果文件夹已经存在，跳过下载
def my_snapshot_download(model_name,cache_dir):
    model_dir = os.path.join(cache_dir, model_name.split('/')[-1])
    if not os.path.exists(model_dir):
        # 如果模型文件夹不存在，执行snapshot download
        model_dir = snapshot_download(model_name, cache_dir=cache_dir)
    else:
        print(f"模型文件夹 {model_dir} 已存在，跳过下载。")
        

cache_dir = '.'
model_name = "AI-ModelScope/bge-small-zh-v1.5"
model_dir=my_snapshot_download(model_name, cache_dir=cache_dir)
# 源大模型下载
model_name = "IEITYuan/Yuan2-2B-July-hf"
model_dir = my_snapshot_download(model_name, cache_dir=cache_dir)

# 定义向量模型类
class EmbeddingModel:
    """
    class for EmbeddingModel
    """

    def __init__(self, path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.model = AutoModel.from_pretrained(path).cuda()
        print(f'Loading EmbeddingModel from {path}.')

    def get_embeddings(self, texts: List) -> List[float]:
        """
        calculate embedding for text list
        """
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()
    
print("> Create embedding model...")
embed_model_path = './AI-ModelScope/bge-small-zh-v1___5'
embed_model = EmbeddingModel(embed_model_path)

# 定义向量库索引类
class VectorStoreIndex:
    """
    class for VectorStoreIndex
    """

    def __init__(self, doecment_path: str, embed_model: EmbeddingModel) -> None:
        self.documents = []
        for line in open(doecment_path, 'r', encoding='utf-8'):
            line = line.strip()
            self.documents.append(line)

        self.embed_model = embed_model
        self.vectors = self.embed_model.get_embeddings(self.documents)

        print(f'Loading {len(self.documents)} documents for {doecment_path}.')

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

    def query(self, question: str, k: int = 1) -> List[str]:
        question_vector = self.embed_model.get_embeddings([question])[0]
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])
        return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist() 
    

# 定义大语言模型类
class LLM:
    """
    class for Yuan2.0 LLM
    """

    def __init__(self, model_path: str) -> None:
        print("Creat tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

        print("Creat model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

        print(f'Loading Yuan2.0 model from {model_path}.')

    def generate(self, question: str, context: List, temperature: float = 1.0):
        if context:
            prompt = f'背景：{context}\n问题：{question}\n请基于背景，回答问题。不超过50个字。'
        else:
            prompt = question

        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        # outputs = self.model.generate(inputs, do_sample=False, max_length=1024)
        outputs = self.model.generate(inputs,do_sample=True, max_length=1024,temperature=temperature)#, top_k=50, top_p=0.95)
        output = self.tokenizer.decode(outputs[0])

        # print(output.split("<sep>")[-1])
        
        return output.split("<sep>")[-1].split("<eod>")[0]
    

print("> Create Yuan2.0 LLM...")
# model_path = './IEITYuan/Yuan2-2B-Mars-hf'
model_path = './IEITYuan/Yuan2-2B-July-hf'
llm = LLM(model_path)

# 微信状态列表和对应知识库
button_values = ['chill', 'crush', 'joy']
true_values=['松弛、自在','恋爱、心动','快乐、美滋滋']
print("> Create index...")
index_list=[]
for value in button_values:
    # doecment_path = './knowledge.txt'
    doecment_path=f'/mnt/workspace/打工嘴替机器人/{value}.txt'
    index_list.append(VectorStoreIndex(doecment_path, embed_model))
    
# 导入前端开发库
import streamlit as st

def main():
# if __name__ == '__main__':
    # 创建一个标题
    st.title('💬 打工人嘴替机器人') 
    st.caption("🚀 A Streamlit chatbot powered by 浪潮信息Yuan2.0大模型")
    
    with st.chat_message("assistant"):
        st.write("嗨👋，我是一个苦命打工人，看来你想设置一个微信状态，需要一个配套文案。你想设置一个什么状态？")
    
    
    for value,index,true_value in zip(button_values,index_list,true_values):
        if st.button(value):
            st.write(f"好的老板，我帮你想一个 {value}的文案")
            
            prompt1=f"生成一句带有'{true_value}'情绪价值的句子，少于10个字。"
            output1=llm.generate(prompt1, [],temperature=1.2)

            print(prompt1)
            print(f"根据prompt1生成的与{true_value}有关的中文句子：\n{output1}")
            
            # 矢量化而且寻找匹配
            context = index.query(output1)
            print(f"context{context}")
            
            prompt2=f"""
            请将以下句子进行简单修改，使其表达出明显的{true_value}情绪：
            原句：{context}
            只输出修改后的句子。
            """
            summary=llm.generate(prompt2,context,temperature=1.2)
            
            print(f"prompt2最终为：\n{prompt2}\n输出的结果最终为：\n{summary}")
            
            st.chat_message("assistant").write(summary)
            
if __name__ == '__main__':
    main()