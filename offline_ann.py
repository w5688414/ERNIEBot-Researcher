import argparse
import os

from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

from tools.utils import FaissSearch, build_index

parser = argparse.ArgumentParser()
parser.add_argument(
    "--embedding_type",
    type=str,
    default="ernie_embedding",
    help="['openai_embedding','ernie_embedding']",
)

args = parser.parse_args()

if __name__ == "__main__":
    access_token = os.environ.get("EB_AGENT_ACCESS_TOKEN", None)
    if args.embedding_type == "openai_embedding":
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada",
            openai_api_version="2023-07-01-preview",
        )
    elif args.embedding_type == "ernie_embedding":
        embeddings = ErnieEmbeddings(aistudio_access_token=access_token)
    else:
        raise NotImplementedError

    db = build_index(
        "faiss_abstract_baidu_openai",
        embeddings=embeddings,
        path="abstract_corpus_baidu_openai.json",
        abstract=True,
    )
    faiss_tool = FaissSearch(db, embeddings)
    print(faiss_tool.search("what is the meaning of life", 1))
