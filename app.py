import streamlit as st
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
import torch
import re
import regex
from nltk import sent_tokenize, word_tokenize
import nltk
from rouge_score import rouge_scorer
import warnings 
from bert_score import score
warnings.filterwarnings("ignore")

nltk.download('punkt')

def tokenized_sent_to_str(tokenized_sent) -> str:
    # remove redundant spaces around sentence delimiters
    for p in ".,?!:;":
        tokenized_sent = tokenized_sent.replace(f" {p}", f"{p}")
    return tokenized_sent

def merge_characters(doc: str):
    # Rule-based merging
    doc_merged = ""
    doc_split = doc.split("\n")
    max_lines = len(doc_split)
    # Line-level operation is safer and more flexible
    # as in some reports only a few lines require character merging
    for i, line in enumerate(doc_split):
        line_new = str(line)
        # If there is a `\ \t \ ` assume tabs are spaces and delete spaces
        if regex.findall(pattern=r"(?<=\w)\ \t\ ", string=line):
            pattern = regex.compile(r"\ ")
            line_new = pattern.sub("", line)
        if i != max_lines - 1:  # control final end-line
            line_new += "\n"
        doc_merged += line_new
    # Swap tabs for single spaces
    doc_merged = doc_merged.replace("\t", " ")
    return doc_merged

def clean(doc: str):
    # Remove upper-cased lines
    doc = "\n".join([l for l in doc.split("\n") if not l.isupper()]).strip()
    # Perform character-level merging when tabs are used as spaces
    doc = "\n".join([merge_characters(l) for l in doc.split("\n")])
    # Make document compact
    doc = doc.replace("\n", " ")
    # remove duplicated spaces
    doc = " ".join(doc.split())
    # reconnect words split by end-of-line hyphenation with lookbehind
    doc = regex.sub(r"(?<=[a-z])-\s", "", doc)
    # remove emails, urls, hours, UK phones, dates
    doc = doc.replace(
        "WWW.", "www."
    )  # ensure url starts lowercase to be caught by regex below
    reg_to_drop = r"""(?x)
        (?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])
        | (https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})
        | ^([0-1]?[0-9]|2[0-3])(:[0-5][0-9])+
        | ^(((\+44\s?\d{4}|\(?0\d{4}\)?)\s?\d{3}\s?\d{3})|((\+44\s?\d{3}|\(?0\d{3}\)?)\s?\d{3}\s?\d{4})|((\+44\s?\d{2}|\(?0\d{2}\)?)\s?\d{4}\s?\d{4}))(\s?\#(\d{4}|\d{3}))?$
        | ^(0?[1-9]|[12][0-9]|3[01])[\/\-](0?[1-9]|1[012])[\/\-]\d{4}$
    """
    pattern = regex.compile(reg_to_drop, regex.UNICODE)
    doc = pattern.sub("", doc)
    # Remove non-alphanumeric and non-special financial characters
    reg_to_drop = r"""(?x) # flag to allow comments and multi-line regex
            [^\w_ |        # alpha-numeric
            \p{Sc} |       # currencies
            \%\&\"\\'\’\(\)\.\,\?\!\-\;\\\/ ]  # preserve apostrophe `’`
        """
    pattern = regex.compile(reg_to_drop, regex.UNICODE)
    doc = pattern.sub("", doc)
    doc = doc.replace("’", "'")  # replace special unicode apostrophe with normal one
    # Deal with unmerged apostrophes
    apostrophes = r"\s*\'s"
    pattern = regex.compile(apostrophes, regex.UNICODE)
    doc = pattern.sub("'s", doc)
    # remove duplicated spaces after dropping special symbols
    doc = " ".join(doc.split())
    # remove redundant spaces around sentence delimiters
    for p in ".,?!:;":
        doc = doc.replace(f" {p}", f"{p}")
    # normalise accents and umlauts
    # doc = unidecode(doc)  # unfortunately normalizes currencies as well
    return doc

def preprocess_text(file):
    lines = file.splitlines()

    cleaned_lines = []
    for line in lines:
        # Check if "fax" or "tel" is present in the line
        if 'fax' in line.lower() or 'tel' in line.lower():
            continue  # Skip the line if "fax" or "tel" is present

        # Remove extra white spaces, except in numbers or dates
        cleaned_line = re.sub(r'\s+', ' ', line.strip())

        # Remove leading/trailing spaces from each line
        cleaned_line = cleaned_line.strip()

        # Ensure spaces around hyphens to retain formatting in numbers or dates
        cleaned_line = re.sub(r'(\d) - (\d)', r'\1-\2', cleaned_line)

        # Remove standalone dashes but retain dashes in words (like non-financial)
        cleaned_line = re.sub(r'\b-\b', '', cleaned_line)

        # Retain specific formatting in numbers and dates
        cleaned_line = re.sub(r'(\d{2}/\d{2}/\d{2})', r'\1 ', cleaned_line)
        cleaned_line = re.sub(r'(\d{2}/\d{2}/\d{2}) to (\d{2}/\d{2}/\d{2})', r'\1 to \2', cleaned_line)

        # Remove currency symbols if they are standalone
        cleaned_line = re.sub(r'(?<!\w)[£$€](?!\w)', '', cleaned_line)

        # Remove repeating special characters if they are at the start or continuously
        cleaned_line = re.sub(r'(\W)\1{0,}', r'\1', cleaned_line)

        # Remove trailing period at the end of sentences
        cleaned_line = re.sub(r'(\w+)\.\s$', r'\1', cleaned_line)

        # Remove empty lines and lines with only whitespace characters
        if cleaned_line.strip():
            # Lowercase the line and add to cleaned lines
            cleaned_lines.append(cleaned_line)

    return '\n'.join(cleaned_lines)

def preprocess(doc: str) -> str:
    # Clean the data
    doc = clean(doc)
    # Split content document into sentences
    cleaned_doc = sent_tokenize(doc)

    doc_tokenized = ""
    for s in cleaned_doc:
        num_numbers = 0
        num_words = 0
        w = word_tokenize(s)
        for t in w:
            if not any(char.isdigit() for char in t):
                num_words += 1
            # Check if the token is a number
            elif t.replace('.', '').isdigit():
                num_numbers += 1

        if len(w) > 3 and num_numbers <= num_words:
            doc_tokenized += s + " "

    return str(doc_tokenized)

def parse_data(data):
    dataset = {'Sentence': []}
    cleaned_text = preprocess_text(data)
    cleaned_text = preprocess(cleaned_text)
    cleaned_tokenizer = sent_tokenize(cleaned_text)
    print(cleaned_tokenizer)
    for sentence in cleaned_tokenizer:
        dataset['Sentence'].append(sentence)

    filtered_sentences = {"Sentenece":[],"sentence_id":[],"Score":[]}
    for i,sentence in enumerate(dataset['Sentence']):
        if count_tokens(sentence) > 5 and not should_remove(sentence):
            filtered_sentences['Sentenece'].append(sentence)
            filtered_sentences['sentence_id'].append(i+1)


    return filtered_sentences

def count_tokens(sentence):
    tokens = word_tokenize(sentence)
    return len(tokens)

def should_remove(sentence):
    num_white_spaces = sum(1 for char in sentence if char.isspace())
    num_non_white_spaces = len(sentence) - num_white_spaces
    return num_white_spaces > (num_non_white_spaces / 3)

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForSequenceClassification.from_pretrained('bert_regression_model_2/', num_labels=1)

st.title("Financial Narrative Summarization")

test_input = st.text_area('Anuual Report')


if st.button("Submit"):
    filtered_sentences = parse_data(test_input)

    test_encodings = tokenizer(filtered_sentences["Sentenece"], truncation=True, padding=True, max_length=128)
    test_encodings_dict = test_encodings.data
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_encodings_dict['input_ids']),
        torch.tensor(test_encodings_dict['attention_mask'])
    )

    # Define custom data collator
    class CustomTestDataCollator(DataCollatorWithPadding):
        def __call__(self, features):
            # Unpack features
            input_ids = torch.stack([torch.tensor(feature[0].clone().detach()) for feature in features])
            attention_mask = torch.stack([torch.tensor(feature[1].clone().detach()) for feature in features])

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

    # Instantiate custom data collator
    test_data_collator = CustomTestDataCollator(tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=128,  # Increase batch size
        per_device_eval_batch_size=128,    # Increase batch size for evaluation
        num_train_epochs=1,               # Reduce number of epochs
        learning_rate=5e-5,               # Increase learning rate
        output_dir="./output",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='mse',
        greater_is_better=False,
        # fp16=True,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        warmup_steps=200,                 # Decrease warmup steps
    )

    # Create a Trainer instance with the saved model and test dataset
    trainer = Trainer(
        model=model,
        data_collator=test_data_collator,
        args=training_args,
        eval_dataset=test_dataset,
    )

    # Evaluate the model on the test set
    print("Predicting.......*")

    with st.spinner('Generating Summary...'):
        # Predict
        predictions = trainer.predict(test_dataset)

    for i in range(len(predictions[0])):
        filtered_sentences["Score"].append(predictions[0][i][0])
        
    sorted_data = sorted(zip(filtered_sentences['Sentenece'], filtered_sentences['sentence_id'], filtered_sentences['Score']), key=lambda x: x[2], reverse=True)

    sorted_dict = {
        'Sentenece': [item[0] for item in sorted_data],
        'sentence_id': [item[1] for item in sorted_data],
        'Score': [item[2] for item in sorted_data]
    }

    output = {
    "Sentence": [],
    "sentence_id": [],
    "Score": []
    }
    summary = ""
    for j in range(len(sorted_dict["Sentenece"])):
        if len(word_tokenize(summary)) < 1000:
            summary += sorted_dict["Sentenece"][j]
            output["Sentence"].append(sorted_dict["Sentenece"][j])
            output["sentence_id"].append(sorted_dict["sentence_id"][j])
            output["Score"].append(sorted_dict["Score"][j])

    sorted_summ = sorted(zip(output['Sentence'], output['sentence_id'], output['Score']), key=lambda x: x[1], reverse=False)

    sorted_sum_dict = {
        'Sentence': [item[0] for item in sorted_summ],
        'sentence_id': [item[1] for item in sorted_summ],
        'Score': [item[2] for item in sorted_summ]
    }


    # Concatenate sentences into a string
    sorted_sentences = ' '.join(sorted_sum_dict['Sentence'])


    # sorted_dict = dict(sorted(outputs.items(), key=lambda item: item[1], reverse=True))

    # summary = ""

    # for i in sorted_dict:
    #     if len(word_tokenize(summary)) < 1000:
    #         summary =summary+""+i
    #     else:
    #         break
    

    # Define the background color for the Markdown text (you can adjust this to match st.info())
    background_color = "#f0f0f0"  # Background color of st.info() with transparency
    # text_color = "rgb(0, 66, 128)"  # Text color of st.info()
    # color: {text_color};

    # Define the Markdown string
    markdown_style = f"""
        <style>
            .info-background {{
                background-color: {background_color};
                padding: 10px;
                border-radius: 5px;
            }}
        </style>
    """

    # Apply the style and display the string using st.markdown()
    st.markdown(markdown_style, unsafe_allow_html=True)
    st.markdown(f'<div class="info-background" style="text-align: justify">{sorted_sentences}</div>', unsafe_allow_html=True)

    st.write("")
    
