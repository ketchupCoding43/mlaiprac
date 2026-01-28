from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer for DialoGPT
model_name = "microsoft/DialoGPT-medium"  # Options: small, medium, large
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function for chatbot interaction
def chatbot_response(prompt, chat_history_ids=None):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")

    # Append to chat history
    bot_input_ids = (
        torch.cat([chat_history_ids, input_ids], dim=-1)
        if chat_history_ids is not None
        else input_ids
    )

    # Generate response using the model
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
    )

    # Decode and return the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Start the chatbot
print("Chatbot: Hello! I am a chatbot. How can I help you today?")
chat_history = None  # To maintain context in conversation

while True:
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye! Have a great day!")
        break

    # Get response from chatbot
    response, chat_history = chatbot_response(user_input, chat_history)
    print(f"Chatbot: {response}")
