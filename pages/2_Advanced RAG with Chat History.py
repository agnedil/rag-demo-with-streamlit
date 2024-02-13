import streamlit as st
from advanced_rag_history import ElevatedRagWithHistory


st.set_page_config(page_title="Advanced RAG with Chat History", page_icon="🔎")
st.sidebar.header("Advanced RAG with Chat History")
st.session_state['current_page'] = 'page2'


def get_elevated_rag_chain_history() -> ElevatedRagWithHistory:
    '''
    Retrieve an existing ElevatedRagWithHistory instance stored in the session state or create a new one
    Returns:
        ElevatedRagWithHistory: the ElevatedRagWithHistory instance for the current session.
    '''
    # check if instance exists in session state; if not, create one
    if 'elevated_rag_chain_history' not in st.session_state:
        st.session_state.elevated_rag_chain_history = ElevatedRagWithHistory()
    return st.session_state.elevated_rag_chain_history


def handle_query_submission_2() -> None:
    '''
    Handle submission of user query:
    * validate query
    * perform query operation
    * update session state with response to query
    * display error message if query is empty or if RAG system encounters an issue
    '''
    # validate query is not empty
    if not st.session_state.user_query2.strip():
        st.error("Please enter a non-empty query")
        return
    # perform query operation and update session state with response
    try:    
        with st.spinner("Querying the Llama RAG system ..."):
            elevated_rag_chain_history = get_elevated_rag_chain_history()
            response = elevated_rag_chain_history.invoke(st.session_state.user_query2)
            st.session_state.response2 = response
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
    * #ADD8E6 - light blue
    * #CCE0F5 - light cornflower blue
    '''
    custom_css = """
    <style>
    /* Target the root html and body elements along with the main Streamlit container classes */
    html, body, [class*="ViewContainer"], [class*="stApp"]  {
        background-color: #CFEAF5;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)    # inject custom CSS


 # render the Streamlit app interface
def main() -> None:
    
    change_background_color()
    st.title("Query your own data")
    st.markdown(
        '''
        # **Llama 2 RAG w/Chat History**
        * Type in one or more URLs for PDF files - one per line.
        * Click on `Load PDFs` and wait until the RAG system is built.
        * Type your query and click on `Submit Query`.
        * Once the LLM sends back a reponse, it will be displayed in the Reponse field.
        * This system is best suited for cases when subsequent user queries may be related to previous ones.
        
        **Refresh the page to clear / reset the RAG system**
        '''
    )
    st.text('')
    st.text('')

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
                    elevated_rag_chain_history = get_elevated_rag_chain_history()
                    elevated_rag_chain_history.add_pdfs_to_vectore_store(pdf_urls_list)
                    st.success("PDFs loaded successfully!")
            except ValueError:
                st.warning("Could not load PDFs. Make sure your PDF URLs are valid.")

    # input and submit user query
    user_query2 = st.text_input("Enter your query:", key="user_query2")
    submit_query_button2 = st.button("Submit Query", on_click=handle_query_submission_2)

    # initialize or display response field
    if 'response2' not in st.session_state:
        st.session_state['response2'] = ''
    dynamic_height = calculate_text_area_height(st.session_state['response2'])
    if st.session_state['current_page'] == 'page2':
        st.text_area("Response:", value=st.session_state['response2'], height=dynamic_height, key="response_field_page2")


if __name__ == "__main__":
    main()
    