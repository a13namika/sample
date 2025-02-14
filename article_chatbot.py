import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load GPT-2 Model
gpt2_model = "gpt2"
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model)
gpt2_pipeline = pipeline("text-generation", model=gpt2_model, tokenizer=gpt2_tokenizer)

# Load Mistral 7B (requires transformers with Flash Attention)
mistral_model = "mistralai/Mistral-7B-Instruct-v0.1"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model)
mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model, torch_dtype=torch.float16, device_map="auto")

# Load LLaMA 2 7B (Meta's Model)
llama_model = "meta-llama/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model, torch_dtype=torch.float16, device_map="auto")

# Generate an article using each model
def generate_article(prompt, model_type="gpt2"):
    """Generates an article based on the model type."""
    if model_type == "gpt2":
        response = gpt2_pipeline(prompt, max_length=500, num_return_sequences=1)
        return response[0]["generated_text"]
    
    elif model_type == "mistral":
        inputs = mistral_tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = mistral_model.generate(**inputs, max_length=500)
        return mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    elif model_type == "llama":
        inputs = llama_tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = llama_model.generate(**inputs, max_length=500)
        return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main function
if __name__ == "__main__":
    topic = input("Enter a topic for the article: ")

    print("\n=== GPT-2 Generated Article ===\n")
    print(generate_article(topic, model_type="gpt2"))

    print("\n=== Mistral 7B Generated Article ===\n")
    print(generate_article(topic, model_type="mistral"))

    print("\n=== LLaMA 2 Generated Article ===\n")
    print(generate_article(topic, model_type="llama"))
