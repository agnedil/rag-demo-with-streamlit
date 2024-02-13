import streamlit as st
from advanced_rag import ElevatedRagChain


st.set_page_config(page_title="Advanced RAG", page_icon="🔍")
st.sidebar.header("Advanced RAG")


def get_elevated_rag_chain() -> ElevatedRagChain:
    '''
    Retrieve an existing ElevatedRagChain instance stored in the session state or create a new one
    Returns:
        ElevatedRagChain: The ElevatedRagChain instance for the current session.
    '''
    # check if instance exists in session state; if not, create one
    if 'elevated_rag_chain' not in st.session_state:
        st.session_state.elevated_rag_chain = ElevatedRagChain()
    return st.session_state.elevated_rag_chain


def handle_query_submission() -> None:
    '''
    Handle submission of user query:
    * validate query
    * perform query operation
    * update session state with response to query
    * display error message if query is empty or if RAG system encounters an issue
    '''
    # validate query is not empty
    if not st.session_state.user_query.strip():
        st.error("Please enter a non-empty query")
        return
    # perform query operation and update session state with response
    try:    
        with st.spinner("Querying the Llama RAG system ..."):
            elevated_rag_chain = get_elevated_rag_chain()
            response = elevated_rag_chain.elevated_rag_chain.invoke(st.session_state.user_query)
            st.session_state.response = response
    # handle cases where RAG system is not initialized correctly
    except AttributeError:
        st.error("RAG system was not built correctly. Please re-load PDFs.")


def calculate_text_area_height(
        text: str,
        chars_per_line: int = 70,
        default_height: int = 100,
        line_height: int = 27,
        ) -> int:
    '''
    Calculate dynamic height of text area widget based on its content.
    Args:
        text (str): text content to display in the text area.
        chars_per_line (int): estimated number of characters per line, defaults to 50.
        default_height (int): minimum height of the text area, defaults to 100.
        line_height (int): height per line of text, defaults to 27.
    Returns:
        int: calculated height of text area
    '''
    lines = text.count('\n') + 1    # count number of lines in text
    estimated_lines = max(lines, len(text)/chars_per_line)    # estimate based on content length
    return max(default_height, int(estimated_lines * line_height))    # calculate dynamic height


def change_background_color() -> None:
    '''
    Inject custom CSS to change the background color of the Streamlit app
    Favorite colors:
    * #ADD8E6, #BEEEFF, #CEEFFF, #DEFFFF - light blue, 
    * #CCE0F5 - light cornflower blue
    '''
    custom_css = """
    <style>
    /* Target the root html and body elements along with the main Streamlit container classes */
    html, body, [class*="ViewContainer"], [class*="stApp"]  {
        background-color: #DEEEFF;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)    # inject custom CSS


 # render the Streamlit app interface
def main() -> None:
    
    change_background_color()
    st.title("Query your own data")
    st.markdown("# **Llama 2 RAG**")
    st.markdown(
        '''
        **This system is best suited for cases when user queries are not related to each other**  
        **Refresh the page to clear / reset the RAG system**
        '''
    )

    # input area for PDF URLs
    pdf_urls = st.text_area("Enter PDF URLs (one per line):", height=100)
    load_pdfs_button = st.button("Load PDFs")

    # load PDFs into vector store
    if load_pdfs_button:
        if not pdf_urls.strip(): 
            st.warning("Please enter one or more PDF URLs before loading.")
        else:
            try:
                with st.spinner("Loading..."):
                    pdf_urls_list = pdf_urls.split("\n")
                    elevated_rag_chain = get_elevated_rag_chain()
                    elevated_rag_chain.add_pdfs_to_vectore_store(pdf_urls_list)
                    st.success("PDFs loaded successfully!")
            except ValueError:
                st.warning("Could not load PDFs. Make sure your PDF URLs are valid.")

    # input and submit user query
    user_query = st.text_input("Enter your query:", key="user_query")
    submit_query_button = st.button("Submit Query", on_click=handle_query_submission)

    # initialize or display response field
    if 'response' not in st.session_state:
        st.session_state['response'] = ''
    dynamic_height = calculate_text_area_height(st.session_state['response'])
    st.text_area("Response:", value=st.session_state['response'], height=dynamic_height, key="response_field_page1")


if __name__ == "__main__":
    main()