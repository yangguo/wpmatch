import json
import os

import faiss
import pandas as pd
import pinecone

# from gpt_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader
from langchain.chains import RetrievalQA, VectorDBQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceHubEmbeddings,
    OpenAIEmbeddings,
)

# from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAIChat
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS, Chroma, Pinecone, Qdrant

# from qdrant_client import QdrantClient
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()

embeddings = HuggingFaceHubEmbeddings(
    repo_id=model_name,
    task="feature-extraction",
    huggingfacehub_api_token="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",
)

# read config from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# get openai api key from config.json
api_key = config["openai_api_key"]

PINECONE_API_KEY = "515f071e-32d7-4819-8ca5-552c98718605"
PINECONE_API_ENV = "us-west1-gcp"

# initialize pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)


qdrant_host = "127.0.0.1"
# qdrant_api_key = ""


os.environ["OPENAI_API_KEY"] = api_key

uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "fileidx"
backendurl = "http://localhost:8000"

# openai_api_key = os.environ.get("OPENAI_API_KEY")
# if openai_api_key is None:
#     print("请设置OPENAI_API_KEY")
# else:
#     print("已设置OPENAI_API_KEY" + openai_api_key)

# initialize pinecone
# pinecone.init(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_API_ENV
# )


def build_index():
    """
    Ingests data into LangChain by creating an FAISS index of OpenAI embeddings for text files in a folder "fileraw".
    The created index is saved to a file in the folder "fileidx".
    """

    # Get paths to text files in the "fileraw" folder
    # ps = list(Path(filerawfolder).glob("**/*.txt"))

    # data = []
    # sources = []
    # for p in ps:
    #     with open(p) as f:
    #         data.append(f.read())
    #     sources.append(p)

    # # Split the documents into smaller chunks as needed due to the context limits of the LLMs
    # docs = []
    # for d in data:
    #     splits = split_text(d, chunk_chars=800, overlap=50)
    #     docs.extend(splits)

    loader = DirectoryLoader(filerawfolder, glob="**/*.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # use tiktoken
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    # embeddings = OpenAIEmbeddings()
    # Create vector store from documents and save to disk
    # store = FAISS.from_texts(docs, OpenAIEmbeddings())
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(fileidxfolder)

    # use qdrant
    # collection_name = "filedocs"
    # # Create vector store from documents and save to qdrant
    # Qdrant.from_documents(docs, embeddings, host=qdrant_host, prefer_grpc=True, collection_name=collection_name)

    # use pinecone
    # Create vector store from documents and save to pinecone
    # index_name = "langchain1"
    # docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    # return docsearch


# create function to add new documents to the index
def add_to_index():
    """
    Adds new documents to the LangChain index by creating an FAISS index of OpenAI embeddings for text files in a folder "fileraw".
    The created index is saved to a file in the folder "fileidx".
    """

    loader = DirectoryLoader(filerawfolder, glob="**/*.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # use tiktoken
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    # print("docs",docs)
    # get faiss client
    store = FAISS.load_local(fileidxfolder, embeddings)

    # get qdrant client
    # qdrant_client = QdrantClient(host=qdrant_host, prefer_grpc=True)
    # collection_name = "filedocs"
    # # get qdrant docsearch
    # store = Qdrant(qdrant_client, collection_name=collection_name, embedding_function=OpenAIEmbeddings().embed_query)

    # Create vector store from documents and save to disk
    store.add_documents(docs)
    store.save_local(fileidxfolder)


def gpt_vectoranswer(question, chaintype="stuff", top_k=4, model_name="gpt-3.5-turbo"):
    # get faiss client
    store = FAISS.load_local(fileidxfolder, embeddings)

    # pinecone_namespace = "bank"
    # pinecone_index_name = "ruledb"

    # # Create an index object
    # index = pinecone.Index(index_name=pinecone_index_name)

    # index_stats_response = index.describe_index_stats()
    # print(index_stats_response)

    # collection_description = pinecone.describe_index('ruledb')
    # print(collection_description)

    system_template = """根据提供的背景信息，请准确和全面地回答用户的问题。
    如果您不确定或不知道答案，请直接说明您不知道，避免编造任何信息。
    {context}"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(model_name=model_name)
    # chain = VectorDBQA.from_chain_type(
    receiver = store.as_retriever()
    receiver.search_kwargs["k"] = top_k
    chain = RetrievalQA.from_chain_type(
        llm,
        chain_type=chaintype,
        # vectorstore=store,
        retriever=receiver,
        # k=top_k,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    result = chain({"query": question})

    answer = result["result"]
    # sourcedf=None
    source = result["source_documents"]
    sourcedf = docs_to_df_audit(source)

    return answer, sourcedf


def gpt_auditanswer(question, chaintype="stuff", top_k=4, model_name="gpt-3.5-turbo"):
    # get faiss client
    store = FAISS.load_local(fileidxfolder, embeddings)

    system_template = "您是一位资深的 IT 咨询顾问，专业解决问题并能有条理地分析问题。"

    human_template = """
请根据以下政策文件，检查它们整体上是否符合 {question} 的要求。请在回答中描述您的审核过程、依据和推理。
请指出不符合规定的地方，给出改进意见和具体建议。
待审核的监管要求包括：{context}

如果您无法确定答案，请直接回答“不确定”或“不符合”，切勿编造答案。
"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(model_name=model_name)
    # chain = VectorDBQA.from_chain_type(
    receiver = store.as_retriever()
    receiver.search_kwargs["k"] = top_k
    chain = RetrievalQA.from_chain_type(
        llm,
        chain_type=chaintype,
        # vectorstore=store,
        retriever=receiver,
        # k=top_k,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    result = chain({"query": question})

    answer = result["result"]
    # sourcedf=None
    source = result["source_documents"]
    sourcedf = docs_to_df_audit(source)

    return answer, sourcedf


# convert document list to pandas dataframe
def docs_to_df_audit(docs):
    """
    Converts a list of documents to a pandas dataframe.
    """
    data = []
    for document in docs:
        page_content = document.page_content
        metadata = document.metadata
        plc = metadata["source"]
        row = {"内容": page_content, "来源": plc}
        data.append(row)
    df = pd.DataFrame(data)
    return df
