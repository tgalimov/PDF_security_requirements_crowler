import io
import logging
import os
from os import listdir
from os.path import isfile, join
from typing import List, Optional
from link_extractor import download_files

import pandas as pd
import requests
import spacy
# import streamlit as st
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
)

from constants import (
    MODEL_FOLDER, MODEL_PATH, MODEL_TYPE, 
    SEC_LABEL, NONSEC_LABEL, CONFIG_URL,
    PT_PATH, CONFIG_PATH, PT_URL,
)   
from pdf_processing import filter_line, retrieve_lines_from_pdf_file, preprocess
from predict import predict, process_predictions


def download_file_and_save(url, path):
    response = requests.get(url)
    with open(path, "wb+") as f:
        f.write(response.content)


# @st.cache
def download_model():
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    if os.path.exists(MODEL_PATH):
       return

    os.mkdir(MODEL_PATH)
    download_file_and_save(CONFIG_URL, CONFIG_PATH)        
    download_file_and_save(PT_URL, PT_PATH)        


# @st.cache()
def load_model():
    download_model()
    tokenizer = T5Tokenizer.from_pretrained(MODEL_TYPE)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    return model, tokenizer


def load_spacy_model(name="en_core_web_sm"):
    if not spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download('en_core_web_sm')
    nlp = spacy.load(name)
    return nlp


# def set_header():
#     st.markdown("# Security Requirements Extraction")
#     st.markdown("**By Vyacheslav Yastrebov**")


def retrieve_sentences_from_lines(lines: List[str]) -> List[str]:
    nlp = load_spacy_model()
    extracted_sentences = []
    for line in lines:
        doc = nlp(line)
        sentences = [subsentence.strip() for sentence in doc.sents for subsentence in sentence.text.split('\n')]
        relevant_sentences = filter(filter_line, sentences)
        extracted_sentences.extend(relevant_sentences)
    filtered_sentences = map(preprocess, extracted_sentences)
    filtered_sentences = list(set(filtered_sentences))
    return filtered_sentences


def retrieve_security_relevant_sentences(sentences) -> List[str]:
    dataframe = pd.DataFrame(sentences, columns=["Text"])
    classification_model, tokenizer = load_model()
    raw_predictions = predict(classification_model, dataframe, tokenizer, True)
    predictions_dataframe = process_predictions(tokenizer, raw_predictions, dataframe)
    security_dataframe = predictions_dataframe[predictions_dataframe["Label"] == SEC_LABEL]
    security_related_sentences = security_dataframe["Text"].tolist()
    return security_related_sentences


def process_file(file_buffer: Optional[io.BytesIO]=None) -> List[str]:
    lines = retrieve_lines_from_pdf_file(file_buffer)
    sentences = retrieve_sentences_from_lines(lines)
    security_relevant_sentences = retrieve_security_relevant_sentences(sentences)
    return security_relevant_sentences


def show_extracted_sentences(sentences: List[str], file):
    if not sentences:
        # st.markdown("### No security-relevant sentences :(")
        sentence = "### No security-relevant sentences :("
        pass
    # st.markdown("### Extracted security-relevant sentences")
    for sentence in sentences:
        # st.markdown(f"* {sentence}")
        with open("./output/" + file.replace("pdf", "txt"), "a") as f:
            f.write(sentence + "\n")

def main():
    # set_header()
    with open("url_list.txt", "r") as f:
        url_list = f.read().split("\n")
    for url in url_list:
        download_files(url)
        files_dir = "./temp/"
        files_list = [f for f in listdir(files_dir) if isfile(join(files_dir, f))]
        for file in files_list:
            with open(files_dir + file, "rb") as f:
                try:
                    uploaded_file = io.BytesIO(f.read()) #st.file_uploader("Choose a file", type=['pdf'])
                    if uploaded_file is not None:
                        security_requirements = process_file(uploaded_file)
                        show_extracted_sentences(security_requirements, file)
                except Exception as e:
                    print(f"skipped file: " + file)
                    # logging.exception(e)




if __name__ == "__main__":
    main()