from datasets import load_dataset
dataset = load_dataset("tatsu-lab/alpaca")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
def preprocess_data(examples):
    instruction = examples["instruction"]
    response = examples["output"]
    prompt = f"Instruction: {instruction}\nResponse: {response}"
    encodings = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    encodings["labels"] = encodings["input_ids"].copy()  
    return encodings

tokenized_dataset = dataset.map(preprocess_data, remove_columns=["instruction", "output", "input"])

from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  
    r=8,  
    lora_alpha=32,  
    target_modules=["c_attn"],  
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  

from transformers import TrainingArguments, Trainer

train_dataset = tokenized_dataset["train"].select(range(int(len(tokenized_dataset["train"]) * 0.9)))  # 90% for training
eval_dataset = tokenized_dataset["train"].select(range(int(len(tokenized_dataset["train"]) * 0.9), len(tokenized_dataset["train"])))  # 10% for evaluation

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=8,  
    save_total_limit=2,
    save_strategy="epoch",
    fp16=True,  
    logging_dir="./logs",
    logging_steps=10
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  
    eval_dataset=eval_dataset,  
    tokenizer=tokenizer
)

# Train the model
trainer.train()