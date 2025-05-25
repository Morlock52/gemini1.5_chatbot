import os
import time
from typing import List

import streamlit as st
from pypdf import PdfReader

import google.generativeai as genai
from llm_provider import LLMProvider


MEDIA_PATH = os.environ.get("MEDIA_PATH", ".")


def page_setup() -> None:
    st.header("Chat with different types of media/files!", anchor=False, divider="blue")
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


def get_media_type() -> str:
    st.sidebar.header("Select type of Media", divider='orange')
    return st.sidebar.radio(
        "Choose one:",
        ("PDF files", "Images", "Video, mp4 file", "Audio files"),
    )


def get_llm_options():
    st.sidebar.header("LLM Options", divider='rainbow')
    provider = st.sidebar.radio("Provider", ("google", "vertex", "openai"))
    model = st.sidebar.text_input("Model name", "gemini-1.5-pro")
    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 1.0, 0.25)
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.94, 0.01)
    max_tokens = st.sidebar.slider("Maximum Tokens", 100, 5000, 2000, 100)
    return provider, model, temperature, top_p, max_tokens


def _extract_text_from_pdfs(uploaded_files) -> str:
    text = ""
    for pdf in uploaded_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


def handle_pdf_chat(client: LLMProvider) -> None:
    uploaded_files = st.file_uploader("Choose 1 or more PDF", type='pdf', accept_multiple_files=True)
    if not uploaded_files:
        return
    text = _extract_text_from_pdfs(uploaded_files)
    st.write(client.model.count_tokens(text) if hasattr(client.model, 'count_tokens') else '')
    question = st.text_input("Enter your question and hit return.")
    if question:
        response = client.generate_content([question, text])
        st.markdown(response)


# The remaining handlers mirror the original implementation but reuse the LLMProvider

def handle_image_chat(client: LLMProvider) -> None:
    image_file_name = st.file_uploader("Upload your image file.")
    if not image_file_name:
        return
    # Path management is left to the user
    fpath = image_file_name.name
    fpath2 = os.path.join(MEDIA_PATH, fpath)
    if client.provider == 'openai':
        st.warning('Image support for OpenAI is not implemented in this demo.')
        return
    uploaded = genai.upload_file(path=fpath2) if client.provider in {'google', 'vertex'} else None
    while uploaded and uploaded.state.name == "PROCESSING":
        time.sleep(10)
        uploaded = genai.get_file(uploaded.name)
    if uploaded and uploaded.state.name == "FAILED":
        raise ValueError(uploaded.state.name)
    prompt2 = st.text_input("Enter your prompt.")
    if prompt2:
        payload: List = [uploaded, prompt2] if uploaded else [prompt2]
        response = client.generate_content(payload)
        st.markdown(response)
        if uploaded:
            genai.delete_file(uploaded.name)


def main() -> None:
    page_setup()
    media_type = get_media_type()
    provider, model_name, temperature, top_p, max_tokens = get_llm_options()
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
    }
    client = LLMProvider(provider, model_name, generation_config=generation_config,
                         temperature=temperature, max_tokens=max_tokens)
    if media_type == "PDF files":
        handle_pdf_chat(client)
    elif media_type == "Images":
        handle_image_chat(client)
    else:
        st.info("Media type not yet supported in this demo.")


if __name__ == '__main__':
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY_NEW')
    if GOOGLE_API_KEY:
        os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    main()
