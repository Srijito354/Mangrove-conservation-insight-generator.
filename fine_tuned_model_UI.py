import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#declaring the device
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
tuned_model = GPT2LMHeadModel.from_pretrained('D:/VSCode files/Nvidia dev contest/mangrove_finetuned_DialoGPT')
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')

tuned_model.to(dev)

# Create a text input box for the user to enter their text
input_text = st.text_input("Enter your question with respect to freshwater conservation using mangroves: ")

if input_text:
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    input_ids = input_ids.to(next(tuned_model.parameters()).device)

    # Generate response using the model
    output = tuned_model.generate(input_ids, max_length = 1000)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Display the model's response
    st.write(output_text)
