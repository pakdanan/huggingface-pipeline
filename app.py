import streamlit as st
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

@st.cache_resource  # 👈 Add the caching decorator
def load_sentiment_analysis_model():
    return pipeline(
    "sentiment-analysis",
    model="w11wo/indonesian-roberta-base-sentiment-classifier",
    tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier",
    return_all_scores=True
    )    
    
@st.cache_resource  # 👈 Add the caching decorator
def load_ner_model():
    return pipeline(
    "ner",
    model="cahya/bert-base-indonesian-NER",
    tokenizer="cahya/bert-base-indonesian-NER",
    grouped_entities=True
    )

@st.cache_resource  # 👈 Add the caching decorator
def load_T5Tokenizer():
    return T5Tokenizer.from_pretrained("panggi/t5-base-indonesian-summarization-cased")

@st.cache_resource  # 👈 Add the caching decorator
def load_T5Model():
    return T5ForConditionalGeneration.from_pretrained("panggi/t5-base-indonesian-summarization-cased")

def summarize_T5(text):
    tokenizer = load_T5Tokenizer()    #T5Tokenizer.from_pretrained("panggi/t5-base-indonesian-summarization-cased")
    model =  load_T5Model()    #T5ForConditionalGeneration.from_pretrained("panggi/t5-base-indonesian-summarization-cased")
    # generate summary
    input_ids = tokenizer.encode(text, return_tensors='pt')
    summary_ids = model.generate(input_ids,
            max_length=220, 
            num_beams=2,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True,
            no_repeat_ngram_size=2,
            use_cache=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

option = st.selectbox(
    "Select an Option",
    [
        "Classify Text (w11wo/indonesian-roberta-base-sentiment-classifier)",
        "Question Answering (default model)",
        "Text Generation (default model)",
        "Named Entity Recognition (cahya/bert-base-indonesian-NER)",
        "Summarization (panggi/t5-base-indonesian-summarization-cased)",
        "Translation (default model)",
    ],
)

if option == "Classify Text (w11wo/indonesian-roberta-base-sentiment-classifier)":
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
elif option == "Summarization (panggi/t5-base-indonesian-summarization-cased)":
    #summarizer = pipeline("summarization")
    article = st.text_area(label="Paste Article")
    if st.button("Summarize"):
        summary = summarize_T5(article) #summarizer(article, max_length=400, min_length=30)
        st.write(summary)
elif option == "Translation (default model)":
    translator = pipeline("translation_en_to_de")
    text = st.text_area(label="Enter text")
    if st.button("Translate"):
        translation = translator(text)
        st.write(translation)


