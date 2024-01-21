from langchain.embeddings.openai import OpenAIEmbeddings
from tools.utils import build_index, FaissSearch


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada")
    
    db = build_index(
        "faiss_abstract_baidu_openai",
        embeddings=embeddings,
        path="abstract_corpus_baidu_openai.json",
        abstract=True,
    )
    faiss_tool = FaissSearch(db, embeddings)
    print(faiss_tool.search("what is the meaning of life", 1))