import streamlit as st
from transformers import pipeline

option = st.selectbox(
    "Select an Option",
    [
        "Classify Text (default model)",
        "Question Answering (default model)",
        "Text Generation (default model)",
        "Named Entity Recognition (cahya/xlm-roberta-large-indonesian-NER)",
        "Summarization (default model)",
        "Translation (default model)",
    ],
)
button = st.button("Predict")
if option == "Classify Text":
    text = st.text_area(label="Enter text")
    if button:
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)
elif option == "Question Answering":
    q_a = pipeline("question-answering")
    context = st.text_area(label="Enter context")
    question = st.text_area(label="Enter question")
    if button:
        answer = q_a({"question": question, "context": context})
        st.write(answer)
elif option == "Text Generation":
    text = st.text_area(label="Enter text")
    if button:
        text_generator = pipeline("text-generation")
        answer = text_generator(text)
        st.write(answer)
elif option == "Named Entity Recognition":
    text = st.text_area(label="Enter text")
    if button:
        ner = pipeline("token-classification", model="cahya/xlm-roberta-large-indonesian-NER")
        answer = ner(text)
        st.write(answer)
elif option == "Summarization":
    summarizer = pipeline("summarization")
    article = st.text_area(label="Paste Article")
    if button:
        summary = summarizer(article, max_length=400, min_length=30)
        st.write(summary)
elif option == "Translation":
    translator = pipeline("translation_en_to_de")
    text = st.text_area(label="Enter text")
    if button:
        translation = translator(text)
        st.write(translation)
