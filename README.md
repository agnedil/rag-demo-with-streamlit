# Advanced RAG Systems

This repository contains the code for a multi-page web app that demoes a separate Retrieval-Augmented Generation (RAG) system on each page. The system is built using Streamlit, allowing for a user-friendly web interface where users can query their own data loaded from PDFs.

## Features

- **Dynamic Query Processing**: Users can submit queries which are processed in real-time, leveraging various RAG systems for enhanced retrieval and generation.
- **PDF Integration**: The system allows for the loading of PDF documents into a vector store, enabling the RAG system to retrieve information from a vast corpus.
- **Customizable UI**: Includes functions to change elements of the Streamlit app's UI, enhancing the user experience.
- **Error Handling**: Provides feedback on empty queries or issues with the RAG system initialization.

## Installation

To run this application, you need to have Python and Streamlit installed. Follow these steps:

1. Clone this repository to your local machine.
2. Create and activate a virtual environment.
3. Install dependencies from the requirements.txt file by running `pip install -r requirements.txt`.
4. Start the Streamlit app by running `streamlit run `Hello.py`.

## Licence
MIT license
