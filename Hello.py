import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Llama RAG Demo! :raised_hands:")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This is a demo where you can try various RAG apps using the Llama 2 model.  
    **👈 Select a page from the sidebar** to try the corresponding RAG method in practice!
    ### Advanced RAG
    - Provide URLs for PDF files of your choice, and then ask specific questions about them.
    - Elevated RAG algorithm: uses an ensemble retriever with FAISS and BM25 retrievers with re-ranking.
    - Introduced at a [FourthBrainAI.com webinar](https://www.youtube.com/watch?v=XXnc55zypU0&ab_channel=FourthBrainAI).
    
    ### Advanced RAG with Chat History
    - A version of the same advanced RAG algorithm modified to use the chat history.
    - This is a combination of a [Llama recipe](https://github.com/facebookresearch/llama-recipes/blob/main/examples/Getting_to_know_Llama.ipynb) and the FourthBrainAI.com approach.
"""
)