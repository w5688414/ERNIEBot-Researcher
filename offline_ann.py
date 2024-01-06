from langchain.embeddings.openai import OpenAIEmbeddings
from tools.utils import build_index

embeddings = OpenAIEmbeddings(deployment="text-embedding-ada")
db = build_index(
    "faiss_abstract_baidu_openai",
    embeddings=embeddings,
    path="abstract_corpus_baidu_openai.json",
    abstract=True,
)
