
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import toml
#
#
key = 'sk-Sr1286CInNN1FxVqTOzu' + 'T3BlbkFJi1eo1gRQED5dSj4KXyHn'
'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'
page_title = "Vanti chatBI"
page_icon = ":money_with_wings:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/

st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
primaryColor = toml.load(".streamlit/config.toml")['theme']['primaryColor']
style_description = f"""
    <style>
        div.stButton > button:first-child {{ border: 2px solid {primaryColor}; border-radius:10px 10px 10px 10px; }}
        div.stButton > button:hover {{ background-color: {primaryColor}; color:#000000;}}
        footer {{ visibility: hidden;}}
        # header {{ visibility: hidden;}}
    <style>
"""
st.markdown(style_description, unsafe_allow_html=True)

st.title("pandas-ai streamlit interface")

st.write("A demo interface for [PandasAI](https://github.com/gventuri/pandas-ai)")
st.write(
    "Looking for an example *.csv-file?, check [here](https://gist.github.com/netj/8836201)."
)
with st.sidebar:
    st.image('assets/Images/Vanti - Main Logo@4x copy.png')
    if "openai_key" not in st.session_state:
        with st.form("API key"):
            key = st.text_input("OpenAI Key", value="", type="password")
            if st.form_submit_button("Submit"):
                st.session_state.openai_key = key
                st.session_state.prompt_history = []
                st.session_state.df = None

if "openai_key" in st.session_state:
    if st.session_state.df is None:
        uploaded_file = st.file_uploader(
            "Choose a CSV file. This should be in long format (one datapoint per row).",
            type="csv",
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df

    with st.form("Question"):
        question = st.text_input("Question", value="", type="default")
        submitted = st.form_submit_button("Submit")
        if submitted:
            with st.spinner():
                llm = OpenAI(api_token=st.session_state.openai_key)
                pandas_ai = PandasAI(llm)
                x = pandas_ai.run(st.session_state.df, prompt=question)

                fig = plt.gcf()
                if fig.get_axes():
                    st.pyplot(fig)
                st.write(x)
                st.session_state.prompt_history.append(question)

    if st.session_state.df is not None:
        st.subheader("Current dataframe:")
        st.write(st.session_state.df)

    st.subheader("Prompt history:")
    st.write(st.session_state.prompt_history)


if st.button("Clear"):
    st.session_state.prompt_history = []
    st.session_state.df = None