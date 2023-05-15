import os
import pandas as pd
import streamlit as st
from io import StringIO


from src.modules.chatbot import Chatbot_txt, Chatbot
from src.modules.embedder import Embedder_txt, Embedder


class Utilities:
    @staticmethod
    def load_api_key():
        """
        Loads the OpenAI API key from the .env file or from the user's input
        and returns it
        """
        if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
        else:
            user_api_key = st.sidebar.text_input(
                label="#### Your OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password"
            )
            if user_api_key:
                st.sidebar.success("API key loaded", icon="🚀")
        return user_api_key

    @staticmethod
    def handle_upload_txt():
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader("upload", type="txt", label_visibility="collapsed")
        if uploaded_file is not None:

            def show_user_file(uploaded_file):
                file_container = st.expander("Your TXT file :")
                uploaded_file_content = StringIO(uploaded_file.getvalue().decode("utf-8"))
                string_data = uploaded_file_content.read()
                file_container.write(string_data)

            show_user_file(uploaded_file)
        else:
            st.sidebar.info(
                "👆 Upload your TXT file to get started, "
                # "sample for try : [fishfry-locations.csv](https://drive.google.com/file/d/1TpP3thVnTcDO1_lGSh99EKH2iF3GDE7_/view?usp=sharing)"
            )
            st.session_state["reset_chat"] = True
        return uploaded_file

    @staticmethod
    def handle_upload():
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader("upload", type="csv", label_visibility="collapsed")
        if uploaded_file is not None:

            def show_user_file(uploaded_file):
                file_container = st.expander("Your CSV file :")
                shows = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                file_container.write(shows)

            show_user_file(uploaded_file)
        else:
            st.sidebar.info(
                "👆 Upload your CSV file to get started, "
                "sample for try : [fishfry-locations.csv](https://drive.google.com/file/d/1TpP3thVnTcDO1_lGSh99EKH2iF3GDE7_/view?usp=sharing)"
            )
            st.session_state["reset_chat"] = True
        return uploaded_file

    @staticmethod
    def setup_chatbot_txt(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeds = Embedder_txt()
        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            vectors = embeds.getDocEmbeds(file, uploaded_file.name)
            chatbot = Chatbot(model, temperature, vectors)
        st.session_state["ready"] = True
        return chatbot


    @staticmethod
    def setup_chatbot(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeds = Embedder_txt()
        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            vectors = embeds.getDocEmbeds(file, uploaded_file.name)
            chatbot = Chatbot(model, temperature, vectors)
        st.session_state["ready"] = True
        return chatbot
