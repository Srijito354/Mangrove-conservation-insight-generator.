#This program fine-tunes a gpt-2 model on a custom dataset on mangroves.
#However, it shall primarily be used to convert accelerate inference from the same model 
#using TensorRT-LLM.
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
tuned_model = GPT2LMHeadModel.from_pretrained('D:/VSCode files/Nvidia dev contest/mangrove_finetuned_DialoGPT')
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')
#model = GPT2LMHeadModel.from_pretrained(model_name)

while True:
  # Tokenize input text
  input_text = input("Enter your text: ")
  c = input("You wanna continue the conversation, yet after? (y/n): ")
  input_ids = tokenizer.encode(input_text, return_tensors='pt')
  input_ids = input_ids.to(next(tuned_model.parameters()).device)

  # Generate response using the model
  output = tuned_model.generate(input_ids, max_length = 500)
  output_text = tokenizer.decode(output[0], skip_special_tokens=True)
  print(output_text)

  if c == "y":
    pass

  else:
    print("Thanks! Really aplogetic for any mistakes that I made in the conversation. Would definitely try to improve myself!")
    break