from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import langchain

# Use when verbose=True is not working for the chain
langchain.debug = True

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
    # chain_type="stuff",
    # Takes each document and then runs a prompt extracting what's relevant,
    # but sometimes can make up facts on any excerpts that have no relation.
    # Then it summarizes those extractions in another prompt.
    # chain_type="map_reduce",
    #
    # Similar to MapReduce, but instead of extracting relevant info,
    # it answers the question based on the context and scores it on
    # how fully it answered the user's question. It might still make
    # up facts and give it a score of 100, overriding the other relevant
    # answers. It then finds the highest score and returns that as an answer.
    # chain_type="map_rerank",
    #
    # Runs a series of prompts sequentially. Takes the previous answer, adds
    # additional context and sees if it can refine it and update the answer.
    # Could return an answer saying the added context is not relevant, instead
    # of the refined answer.
    # chain_type="refine",
    #
    # Basic type, stuff the document into the prompt, best type
    chain_type="stuff",
)

result = chain.run("What is an interesting fact about the English language?")

print(result)
