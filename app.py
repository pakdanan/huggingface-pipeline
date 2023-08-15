import streamlit as st
from transformers import pipeline

option = st.selectbox(
    "Select an Option",
    [
        "Classify Text (default model)",
        "Question Answering (default model)",
        "Text Generation (default model)",
        "Named Entity Recognition (cahya/bert-base-indonesian-NER)",
        "Summarization (default model)",
        "Translation (default model)",
    ],
)

if option == "Classify Text (default model)":
    text = st.text_area(label="Enter text")
    if st.button("Classify"):
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)
elif option == "Question Answering (default model)":
    q_a = pipeline("question-answering")
    context = st.text_area(label="Enter context")
    question = st.text_area(label="Enter question")
    if st.button("Answer"):
        answer = q_a({"question": question, "context": context})
        st.write(answer)
elif option == "Text Generation (default model)":
    text = st.text_area(label="Enter text")
    if st.button("Generate"):
        text_generator = pipeline("text-generation")
        answer = text_generator(text)
        st.write(answer)
elif option == "Named Entity Recognition (cahya/bert-base-indonesian-NER)":
    text = st.text_area(label="Enter text")
    if st.button("Recognize"):
        ner = pipeline("token-classification", model="cahya/bert-base-indonesian-NER")
        answer = ner(text)
        st.write(answer)
elif option == "Summarization (default model)":
    summarizer = pipeline("summarization")
    article = st.text_area(label="Paste Article")
    if st.button("Summarize"):
        summary = summarizer(article, max_length=400, min_length=30)
        st.write(summary)
elif option == "Translation (default model)":
    translator = pipeline("translation_en_to_de")
    text = st.text_area(label="Enter text")
    if st.button("Translate"):
        translation = translator(text)
        st.write(translation)
