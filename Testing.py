from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = './fine-tuned-distilgpt2'  

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id  

input_text = "Give three tips for staying healthy."

inputs = tokenizer(input_text, return_tensors="pt", padding="longest", truncation=True, max_length=512)

outputs = model.generate(
    **inputs,  # Pass tokenized inputs to the model
    max_length=150,  # Set to a smaller length to avoid long outputs
    num_return_sequences=1,  # Generate only one response
    no_repeat_ngram_size=2,  # Avoid repetitive phrases
    temperature=0.8,  # Control randomness in generation
    top_p=0.85,  # Nucleus sampling
    top_k=30,    # Top-k sampling for more focused generation
    do_sample=True
)


generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
