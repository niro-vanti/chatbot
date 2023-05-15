import os
from dotenv import load_dotenv
from io import BytesIO
from io import StringIO
import sys
import re
from langchain.agents import create_csv_agent
from src.modules.history import ChatHistory
from src.modules.layout import Layout
from src.modules.utils import Utilities
from src.modules.sidebar import Sidebar
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS


# To be able to update the changes made to modules in localhost,
# you can press the "r" key on the localhost page to refresh and reflect the changes made to the module files.
def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]


history_module = reload_module('src.modules.history')
layout_module = reload_module('src.modules.layout')
utils_module = reload_module('src.modules.utils')
sidebar_module = reload_module('src.modules.sidebar')

ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Utilities = utils_module.Utilities
Sidebar = sidebar_module.Sidebar


def init():
    load_dotenv()
    st.set_page_config(layout="wide", page_icon="💬", page_title="ChatBot-Legger")


def main():
    init()
    layout, sidebar, utils = Layout(), Sidebar(), Utilities()
    sidebar.show_logo('assets/Images/colleen-logo.png')

    layout.show_header_txt()
    user_api_key = utils.load_api_key()


    if not user_api_key:
        layout.show_api_key_missing()
    else:
        os.environ["OPENAI_API_KEY"] = user_api_key
        uploaded_file = utils.handle_upload_txt()

        if uploaded_file:
            history = ChatHistory()
            sidebar.show_options()

            uploaded_file_content = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = uploaded_file_content.read()

            # st.write(string_data)


            try:
                chatbot = utils.setup_chatbot_txt(
                    uploaded_file, st.session_state["model"], st.session_state["temperature"]
                )
                st.session_state["chatbot"] = chatbot

                # agent = create_csv_agent(ChatOpenAI(temperature=0),
                #                          uploaded_file_content,
                #                          verbose=True,
                #                          max_iterations=15)

                # embeddings = OpenAIEmbeddings()
                # vectors = FAISS.from_documents([uploaded_file], embeddings)

                # agent = ConversationalRetrievalChain.from_llm(
                #     llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=user_api_key),
                #     retriever=vectors.as_retriever())
                # st.session_state['agent'] = agent
                st.session_state['agent'] = chatbot

                if st.session_state["ready"]:
                    response_container, prompt_container = st.container(), st.container()

                    with prompt_container:
                        is_ready, user_input = layout.prompt_form()

                        history.initialize(uploaded_file)
                        if st.session_state["reset_chat"]:
                            history.reset(uploaded_file)

                        if is_ready:
                            history.append("user", user_input)
                            output = st.session_state["chatbot"].conversational_chat(user_input)

                            history.append("assistant", output)
                            # old_stdout = sys.stdout
                            # sys.stdout = captured_output = StringIO()
                            # agent_answer = chatbot.run(user_input)
                            # sys.stdout = old_stdout
                            # thoughts = captured_output.getvalue()
                            #
                            # cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                            # cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)
                            #
                            # resp = cleaned_thoughts.split('Thought:')[-1].split('Final Answer')
                            # thought = resp[0]
                            # final_answer = resp[1].split('\n')[0].split(': ')[-1]
                            # agent_answer_clean = '\n'.join([thought, final_answer])
                            # full_answer = '\n'.join([output, agent_answer_clean])
                            # history.append("assistant", full_answer)

                    history.generate_messages(response_container)

                    # if st.session_state["show_csv_agent"]:
                    #     query = st.text_input(
                    #         label="Use CSV agent for precise information about the structure of your csv file",
                    #         placeholder="ex : how many rows in my file ?")
                    #     if query != "":
                    #         old_stdout = sys.stdout
                    #         sys.stdout = captured_output = StringIO()
                    #         agent = create_csv_agent(ChatOpenAI(temperature=0),
                    #                                  uploaded_file_content,
                    #                                  verbose=True,
                    #                                  max_iterations=4)
                    #
                    #
                    #         result = agent.run(query)
                    #
                    #         sys.stdout = old_stdout
                    #         thoughts = captured_output.getvalue()
                    #
                    #         cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                    #         cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)
                    #
                    #         with st.expander("Afficher les pensées de l'agent"):
                    #             st.write(cleaned_thoughts)
                    #
                    #         st.write(result)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    sidebar.about()


if __name__ == "__main__":
    main()
