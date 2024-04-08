from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()
db = Chroma(
    embedding_function=embeddings,
    persist_directory="emb",
)

# db.similarity_search("english language")

# Retriever has a typeclass that different DBs must implement
# Has "get_relevant_documents"
retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    # Basic type, stuff the document into the prompt
    chain_type="stuff",
)

result = chain.run("What is an interesting fact about the English language?")

print(result)
