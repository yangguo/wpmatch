# import json
import os
from operator import itemgetter

# import faiss
import pandas as pd

# import pinecone
from dotenv import load_dotenv
from langchain import hub

# from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.schema import StrOutputParser
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS, SupabaseVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_NAME_16K = os.environ.get("AZURE_DEPLOYMENT_NAME_16K")
AZURE_DEPLOYMENT_NAME_GPT4 = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4")
AZURE_DEPLOYMENT_NAME_GPT4_32K = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_32K")
AZURE_DEPLOYMENT_NAME_GPT4_TURBO = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_TURBO")
AZURE_DEPLOYMENT_NAME_EMBEDDING = os.environ.get("AZURE_DEPLOYMENT_NAME_EMBEDDING")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# from qdrant_client import QdrantClient
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()

embeddings = HuggingFaceHubEmbeddings(
    repo_id=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=HF_API_TOKEN,
)


# convert gpt model name to azure deployment name
gpt_to_deployment = {
    "gpt-35-turbo": AZURE_DEPLOYMENT_NAME,
    "gpt-35-turbo-16k": AZURE_DEPLOYMENT_NAME_16K,
    "gpt-4": AZURE_DEPLOYMENT_NAME_GPT4,
    "gpt-4-32k": AZURE_DEPLOYMENT_NAME_GPT4_32K,
    "gpt-4-turbo": AZURE_DEPLOYMENT_NAME_GPT4_TURBO,
}


# choose chatllm base on model name
def get_chatllm(model_name):
    if model_name == "tongyi":
        llm = ChatTongyi(
            streaming=True,
        )
    elif (
        model_name == "ERNIE-Bot-4"
        or model_name == "ERNIE-Bot-turbo"
        or model_name == "ChatGLM2-6B-32K"
        or model_name == "Yi-34B-Chat"
    ):
        llm = QianfanChatEndpoint(
            model=model_name,
        )
    elif model_name == "gemini-pro":
        llm = ChatGoogleGenerativeAI(
            model=model_name, convert_system_message_to_human=True
        )
    else:
        llm = get_azurellm(model_name)
    return llm


# use azure llm based on model name
def get_azurellm(model_name):
    deployment_name = gpt_to_deployment[model_name]
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_BASE_URL,
        openai_api_version="2023-12-01-preview",
        azure_deployment=deployment_name,
        openai_api_key=AZURE_API_KEY,
    )
    return llm


uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "fileidx"
# backendurl = "http://localhost:8000"


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

    loader = DirectoryLoader(filerawfolder, glob="**/*.*")
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

    loader = DirectoryLoader(filerawfolder, glob="**/*.*")
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

    # system_template = """根据提供的背景信息，请准确和全面地回答用户的问题。
    # 如果您不确定或不知道答案，请直接说明您不知道，避免编造任何信息。
    # {context}"""

    # messages = [
    #     SystemMessagePromptTemplate.from_template(system_template),
    #     HumanMessagePromptTemplate.from_template("{question}"),
    # ]
    # prompt = ChatPromptTemplate.from_messages(messages)

    # chain_type_kwargs = {"prompt": prompt}
    prompt = hub.pull("vyang/gpt_answer")

    llm = get_chatllm(model_name)

    # chain = VectorDBQA.from_chain_type(
    retriever = store.as_retriever()
    retriever.search_kwargs["k"] = top_k

    # chain = RetrievalQA.from_chain_type(
    #     llm,
    #     chain_type=chaintype,
    #     # vectorstore=store,
    #     retriever=receiver,
    #     # k=top_k,
    #     return_source_documents=True,
    #     chain_type_kwargs=chain_type_kwargs,
    # )
    # result = chain({"query": question})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    output_parser = StrOutputParser()

    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | output_parser
    )

    rag_chain_with_source = RunnableParallel(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        # "contents": lambda input: [doc.page_content for doc in input["documents"]],
        "documents": lambda input: [doc for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    }

    result = rag_chain_with_source.invoke(question)

    answer = result["answer"]
    # sourcedf=None
    source = result["documents"]
    sourcedf = docs_to_df_audit(source)

    return answer, sourcedf


def gpt_auditanswer(
    question, upload_choice, chaintype="stuff", top_k=4, model_name="gpt-35-turbo"
):
    # get faiss client
    store = FAISS.load_local(fileidxfolder, embeddings)
    filter = upload_to_dict(upload_choice)

    #     system_template = "您是一位资深的 IT 咨询顾问，专业解决问题并能有条理地分析问题。"

    #     human_template = """
    # 请根据以下政策文件，检查它们整体上是否符合 {question} 的要求。请在回答中描述您的审核过程、依据和推理。
    # 请指出不符合规定的地方，给出改进意见和具体建议。
    # 待审核的监管要求包括：{context}

    # 如果您无法确定答案，请直接回答“不确定”或“不符合”，切勿编造答案。
    # """

    #     messages = [
    #         SystemMessagePromptTemplate.from_template(system_template),
    #         HumanMessagePromptTemplate.from_template(human_template),
    #     ]
    #     prompt = ChatPromptTemplate.from_messages(messages)

    #     chain_type_kwargs = {"prompt": prompt}

    chat_prompt = hub.pull("vyang/get_matchplc")
    llm = get_chatllm(model_name)

    # chain = VectorDBQA.from_chain_type(
    retriever = store.as_retriever(search_kwargs={"k": top_k, "filter": filter})
    # receiver.search_kwargs["k"] = top_k
    # chain = RetrievalQA.from_chain_type(
    #     llm,
    #     chain_type=chaintype,
    #     # vectorstore=store,
    #     retriever=retriever,
    #     # k=top_k,
    #     return_source_documents=True,
    #     chain_type_kwargs=chain_type_kwargs,
    # )
    # result = chain({"query": question})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    output_parser = StrOutputParser()

    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | chat_prompt
        | llm
        | output_parser
    )

    rag_chain_with_source = RunnableParallel(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        # "contents": lambda input: [doc.page_content for doc in input["documents"]],
        "documents": lambda input: [doc for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    }

    result = rag_chain_with_source.invoke(question)

    answer = result["answer"]
    # sourcedf=None
    source = result["documents"]
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


def upload_to_dict(lst):
    if len(lst) == 1:
        return {"source": filerawfolder + "/" + lst[0]}
    else:
        return {}
