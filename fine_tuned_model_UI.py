import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
tuned_model = GPT2LMHeadModel.from_pretrained('D:/VSCode files/Nvidia dev contest/mangrove_finetuned_DialoGPT')
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')

# Create a text input box for the user to enter their text
input_text = st.text_input("Enter your text:")

if input_text:
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    input_ids = input_ids.to(next(tuned_model.parameters()).device)

    # Generate response using the model
    output = tuned_model.generate(input_ids, max_length = 5000)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Display the model's response
    st.write(output_text)
