import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the LLaMA-like model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Example model, adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def chat_with_llama(prompt, max_length=100, temperature=0.7):
    """
    Chat with the LLaMA chatbot.
    :param prompt: The input text or question.
    :param max_length: Maximum length of the response.
    :param temperature: Controls randomness of the response.
    :return: Model's response as a string.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Chat loop
print("LLaMA Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    response = chat_with_llama(user_input)
    print(f"LLaMA: {response}")
