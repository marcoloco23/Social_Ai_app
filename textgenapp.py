# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import operator
import numpy as np
import pandas as pd
import os
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#%%


def get_comments(path):
    comment_df = pd.read_csv(
        path)
    comments = comment_df['comment'].to_list()

    return comments
#%%


def scoring_function(context, out_txt):

    model_input = tokenizer_rand.encode(
        context + "<|endoftext|>" + out_txt, return_tensors="pt")
    result_rand = model_rand(model_input, return_dict=True)

    score_rand = round(torch.sigmoid(
        result_rand.logits).squeeze().item(), 2)

    model_input = tokenizer_machine.encode(
        context + "<|endoftext|>" + out_txt, return_tensors="pt")
    result_machine = model_machine(model_input, return_dict=True)

    score_machine = round(torch.sigmoid(
        result_machine.logits).squeeze().item(), 2)

    score = round(np.mean(
        [score_rand, score_machine]), 2)

    return score
#%%


def get_responses(tokenizer, model, comments, temperature=1.0, max_length=50, samples=5, scoring=False):
    context = ""
    dict_list = []
    for comment in comments:
        input_txt = comment
        context += input_txt + '. '

        new_user_input_ids = tokenizer.encode(
            input_txt + tokenizer.eos_token, return_tensors='pt')

        out_txt_list = []
        score_list = []
        for i in range(samples):

            chat_history_ids = model.generate(
                new_user_input_ids, max_length, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.9,
                temperature=temperature)

            out_txt = tokenizer.decode(
                chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

            if scoring == True:
                score = scoring_function(comment, out_txt)
                score_list.append(score)
            else:
                score_list.append('NoScore_{}'.format(i))

            out_txt_list.append(out_txt)

        dictionary = dict(zip(score_list, out_txt_list))
        dict_list.append(dictionary)

    return dict_list

# %%


def get_response(tokenizer, model, prompt="<|startoftext|>", temperature=1.0, max_length=50, samples=5, scoring=False):

    input_txt = prompt

    new_user_input_ids = tokenizer.encode(
        input_txt + tokenizer.eos_token, return_tensors='pt')

    out_txt_list = []
    score_list = []
    for i in range(samples):
        chat_history_ids = model.generate(
            new_user_input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.9,
            temperature=temperature)

        out_txt = tokenizer.decode(
            chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

        if scoring == True:
            score = scoring_function(prompt, out_txt)
            score_list.append(score)
        else:
            score_list.append('NoScore_{}'.format(i))

        out_txt_list.append(out_txt)
    dictionary = dict(zip(score_list, out_txt_list))

    return dictionary
#%%


def generate_p(tokenizer, model, prompt="<|endoftext|>", temperature=1.0, max_length=50, samples=1):

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

    sample_outputs = model.generate(
        generated,
        bos_token_id=random.randint(1,30000),
        pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.9,
        max_length=max_length,
        num_return_sequences=samples,
        temperature=temperature
    )
    out_txt_list = []
    score_list = []
    for i, sample_output in enumerate(sample_outputs):
        out_txt = tokenizer.decode(
            sample_output, skip_special_tokens=True)
        
        if scoring == True:
            score = scoring_function(prompt, out_txt)
            score_list.append(score)
        else:
            score_list.append('NoScore_{}'.format(i))

        out_txt_list.append(out_txt)
    dictionary = dict(zip(score_list, out_txt_list))

    return dictionary
#%%
def generate_np(tokenizer, model, temperature=1.0, max_length=50, samples=1):

    generated = torch.tensor(tokenizer.encode("<|endoftext|>")).unsqueeze(0)

    sample_outputs = model.generate(
        generated,
        bos_token_id=random.randint(1,30000),
        pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.9,
        max_length=max_length,
        num_return_sequences=samples,
        temperature=temperature
    )
    out_txt_list = []
    score_list = []
    for i, sample_output in enumerate(sample_outputs):
        out_txt = tokenizer.decode(
            sample_output, skip_special_tokens=True)

        if scoring == True:
            score = scoring_function(prompt, out_txt)
            score_list.append(score)
        else:
            score_list.append('NoScore_{}'.format(i))

        out_txt_list.append(out_txt)
    dictionary = dict(zip(score_list, out_txt_list))

    return dictionary
#%%
def generate_lyrics(tokenizer, model, temperature=1.0, max_length=50, samples=1):

    generated = torch.tensor(tokenizer.encode("<|endoftext|>")).unsqueeze(0)

    sample_outputs = model.generate(
        generated,
        bos_token_id=random.randint(1,30000),
        pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.9,
        max_length=max_length,
        num_return_sequences=samples,
        temperature=temperature
    )

    return sample_outputs
# %%
def get_tokenizer_and_model(tokenizer_choice, model_choice):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_choice)
    model = AutoModelForCausalLM.from_pretrained(model_choice)
    return tokenizer, model

#%%
def get_scoring_models():
    model_card_rand = "microsoft/DialogRPT-human-vs-rand"
    tokenizer_rand = AutoTokenizer.from_pretrained(model_card_rand)
    model_rand = AutoModelForSequenceClassification.from_pretrained(
        model_card_rand)

    model_card_machine = "microsoft/DialogRPT-human-vs-machine"
    tokenizer_machine = AutoTokenizer.from_pretrained(
        model_card_machine)
    model_machine = AutoModelForSequenceClassification.from_pretrained(
        model_card_machine)

    return tokenizer_rand, model_rand, tokenizer_machine, model_machine

# %%
st.title('Social AI')

#Tokenizers
DialoGPT_large = 'microsoft/DialoGPT-large'
gpt2_large = "gpt2-large"

#Dialog Models
final_character = 'Models/response_model_1'
movie_dialog = 'Models/response_model_dialog'

#Completion Models
poems = "Models/poetry_model"
insta_model = 'Models/gypsea_model'
twitter_model = 'Models/twitter_model'
lyrics_model = 'Models/lyrics_model'

st.sidebar.header('Choose output type')

platform_choice = st.sidebar.selectbox('Choose Platform', ['Instagram','Twitter'], key='platform')

if platform_choice == 'Instagram':
    tokenizer_choice = gpt2_large
    model_choice = insta_model

if platform_choice == 'Twitter':
    tokenizer_choice = gpt2_large
    model_choice = twitter_model
    st.sidebar.write('Make sure max length is less than 70')

#Parameters and options
temperature = st.sidebar.slider('Temperature', 1.0, 5.0, 1.0, key="temp")
max_length = st.sidebar.slider('Max Length', 10, 100, 10, key="max_length")
samples = st.sidebar.slider('Samples', 1, 10, 1, key="samples")
scoring = st.sidebar.checkbox('Scoring?', key='scoringbox')

prompt = st.sidebar.text_input('Enter a prompt')
caption_p_button = st.sidebar.button(
    'Generate caption with prompt', key='prmptbtn')
caption_np_button = st.sidebar.button(
    'Generate caption without prompt', key='genbtn')
response_button = st.sidebar.button('Get Response to prompt', key='rspbtn')

st.sidebar.subheader('More Options')
poem_button = st.sidebar.button('Generate poem')
lyrics_button = st.sidebar.button('Get rap lyrics')
comment_file = st.sidebar.file_uploader('Upload CSV file of comments')
comment_response_button = st.sidebar.button(
    'Get comment responses from file', key='cmtrspbtn')

if caption_p_button == True:
    if scoring == True:
        with st.spinner(text='Loading Scoring Models'):
            tokenizer_rand, model_rand, tokenizer_machine, model_machine = get_scoring_models()
    with st.spinner(text='Loading Models'):
        tokenizer, model = get_tokenizer_and_model(tokenizer_choice, model_choice)
    with st.spinner(text='Thinking...'):
        dictionary = generate_p(tokenizer, model, prompt,
                        temperature, max_length, samples)
        st.success('Done')
        
    st.write(dictionary)

if caption_np_button == True:
    if scoring == True:
        with st.spinner(text='Loading Scoring Models'):
            tokenizer_rand, model_rand, tokenizer_machine, model_machine = get_scoring_models()
    with st.spinner(text='Loading Models'):
        tokenizer, model = get_tokenizer_and_model(
            tokenizer_choice, model_choice)
    with st.spinner(text='Thinking...'):
        dictionary = generate_np(tokenizer, model,
                                 temperature, max_length, samples)
        st.success('Done')

    st.write(dictionary)

if poem_button == True:
    with st.spinner(text='Loading Models'):
        tokenizer, model = get_tokenizer_and_model(
            gpt2_large, poems)
    with st.spinner(text='Thinking...'):
        dictionary = generate_np(tokenizer, model,
                                temperature, max_length, 1)
        st.success('Done')

    st.write(dictionary)

if lyrics_button == True:
    with st.spinner(text='Loading Models'):
        tokenizer, model = get_tokenizer_and_model(
            gpt2_large, lyrics_model)
    with st.spinner(text='Thinking...'):
        sample_outputs = generate_lyrics(tokenizer, model,
                                 temperature, 400, 1)
        st.success('Done')
    for i, sample_output in enumerate(sample_outputs):
        st.text("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

if response_button == True:
    if scoring == True:
        with st.spinner(text='Loading Scoring Models'):
            tokenizer_rand, model_rand, tokenizer_machine, model_machine = get_scoring_models()
    with st.spinner(text='Loading Models'):
        tokenizer, model = get_tokenizer_and_model(
            DialoGPT_large, movie_dialog)
    with st.spinner(text='Thinking...'):
        dictionary = get_response(tokenizer, model, prompt,
                        temperature, max_length, samples, scoring)
        st.success('Done')

    st.write(dictionary)

if comment_response_button == True:
    try:
        comments = get_comments(comment_file)
        st.header('Comments:')
        st.write(comments)
        if scoring == True:
            with st.spinner(text='Loading Scoring Models'):
                tokenizer_rand, model_rand, tokenizer_machine, model_machine = get_scoring_models()
        with st.spinner(text='Loading Models'):
            tokenizer, model = get_tokenizer_and_model(
                DialoGPT_large, movie_dialog)
        with st.spinner(text='Getting Comment Responses...'):
            dict_list = get_responses(tokenizer, model, comments, temperature,
                                    max_length, samples, scoring)
            st.success('Done')
        st.write(dict_list)
    except:
        st.header('No file found')





