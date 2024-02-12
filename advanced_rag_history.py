import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List
from langchain.llms import Replicate
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.chains import ConversationalRetrievalChain


class ElevatedRagWithHistory:
    '''
    Class ElevatedRagWithHistory integrates various components from the langchain library to build
    an advanced retrieval-augmented generation (RAG) system designed to process documents
    by reading in, chunking, embedding, and adding their chunk embeddings to FAISS vector store
    for efficient retrieval. It uses the embeddings to retrieve relevant document chunks
    in response to user queries.
    The chunks are retrieved using an ensemble retriever (BM25 retriever + FAISS retriver)
    and passed through a Cohere reranker before being used as context
    for generating answers using a Llama 2 large language model (LLM).
    The previous chat history is used to re-phrase the user query in order to levarage
    the past reponses provided by the LLM.
    '''
    def __init__(self) -> None:
        '''
        Initialize the class with predefined model, embedding function, weights, and top_k value
        '''
        self.llama2_70b   = 'meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48'
        self.embed_func   = CohereEmbeddings(model="embed-english-light-v3.0")
        self.bm25_weight  = 0.6
        self.faiss_weight = 0.4
        self.top_k        = 5


    def add_pdfs_to_vectore_store(
            self,
            pdf_links: List,
            chunk_size: int=1500,
            ) -> None:
        '''
        Processes PDF documents by loading, chunking, embedding, and adding them to a FAISS vector store.
        Build an advanced RAG system  
        Args:
            pdf_links (List): list of URLs pointing to the PDF documents to be processed
            chunk_size (int, optional): size of text chunks to split the documents into, defaults to 1500
        '''        
        # load pdfs
        self.raw_data = [ OnlinePDFLoader(doc).load()[0] for doc in pdf_links ]

        # chunk text
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        self.split_data    = self.text_splitter.split_documents(self.raw_data)

        # add chunks to BM25 retriever
        self.bm25_retriever   = BM25Retriever.from_documents(self.split_data)
        self.bm25_retriever.k = self.top_k

        # embed and add chunks to vectore store
        self.vector_store     = FAISS.from_documents(self.split_data, self.embed_func)
        self.faiss_retriever  = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        print("All PDFs processed and added to vectore store.")
        
        # build advanced RAG system
        self.build_elevated_rag_system()
        print("RAG system is built successfully.")


    def build_elevated_rag_system(self) -> None:
        '''
        Build an advanced RAG system from different components:
        * BM25 retriever
        * FAISS vector store retriever
        * Chat history
        * Llama 2 model 
        '''
        # combine BM25 and FAISS retrievers into an ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[self.bm25_weight, self.faiss_weight]
        )

        # use reranker to improve retrieval quality
        self.reranker = CohereRerank(top_n=5)
        self.rerank_retriever = ContextualCompressionRetriever(    # combine ensemble retriever and reranker
            base_retriever=self.ensemble_retriever,
            base_compressor=self.reranker,
        )

        # initialize Llama 2 model with specific parameters
        self.llm = Replicate(
            model=self.llama2_70b,
            model_kwargs={"temperature": 0.5,"top_p": 1, "max_new_tokens":1000}
        )
                       
        # initialize RAG chain with chat history
        self.chat_history = []
        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.rerank_retriever,
            return_source_documents=True,
        )


    def invoke(self, query: str) -> str:
        '''
        Process user query through RAG system, leveraging chat history:
        * take user query as input;
        * passes it along with the accumulated chat history to the ConversationalRetrievalChain;
        * appends query and response to chat history;
        * keep chat history manageable and relevant by retaining only the last 5 interactions.

        Args:
            query (str): user's query to be processed by RAG system.

        Returns:
            str: generated answer to user's query.
        '''
        # process query + chat history through RAG system
        result = self.chain({"question": query, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))    # update chat history
        if len(self.chat_history) > 5:    # keep the most recent 5 turns in chat history
            self.chat_history = self.chat_history[-5:]
        return result['answer']    # return response generated by LLM
        