# Install necessary libraries
# !pip install transformers datasets torch pandas

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import torch

# 1. Load the dataset (you can replace the path with your actual CSV dataset path)
df = pd.read_csv("hf://datasets/KingstarOMEGA/HTML-CSS-UI/HCUI.csv")

# Check the first few rows to see how the dataset is structured (assumed to have a 'css_code' column)
print(df.head())

# Convert the DataFrame to a Hugging Face dataset
dataset = Dataset.from_pandas(df)

# 2. Load the tokenizer and pre-trained model (e.g., LLaMA or Command-A)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1")  # Replace with the correct model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1")  # Replace with the correct model

# 3. Preprocess the dataset (tokenize the CSS code)
def preprocess_function(examples):
    # Assuming the column with CSS code is 'css_code', adjust if necessary
    return tokenizer(examples["css_code"], truncation=True, padding="max_length", max_length=512)

# Apply preprocessing to the entire dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 4. Split the dataset into train and test (validation) sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 5. Set up the data collator and training arguments
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # For causal LM

training_args = TrainingArguments(
    output_dir="./results",             # Output directory for model checkpoints and logs
    evaluation_strategy="epoch",        # Evaluate after every epoch
    learning_rate=2e-5,                 # Learning rate
    per_device_train_batch_size=4,      # Training batch size
    per_device_eval_batch_size=8,       # Evaluation batch size
    num_train_epochs=3,                 # Number of epochs
    weight_decay=0.01,                  # Weight decay
    save_steps=500,                     # Save model checkpoints every 500 steps
    logging_dir="./logs",               # Directory for logging
    logging_steps=10,                   # Log every 10 steps
    save_total_limit=2,                 # Only save the last 2 checkpoints
)

# 6. Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 7. Fine-tune the model
trainer.train()

# 8. Save the trained model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# 9. Load the fine-tuned model and tokenizer for inference
model_path = "./fine_tuned_model"  # Path to the saved model directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 10. Generate explanations for CSS code examples
def generate_explanation(css_code, max_length=150):
    inputs = tokenizer(css_code, return_tensors="pt")
    
    # Move model and inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    output = model.generate(
        inputs['input_ids'], 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        top_p=0.95, 
        top_k=50, 
        temperature=1.0, 
        do_sample=True
    )
    
    # Decode and return the generated text
    generated_explanation = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_explanation

# 11. Example usage of the fine-tuned model
# Assuming 'css_code' column contains the CSS examples in your dataset
css_example = df["css_code"].iloc[0]  # Taking the first CSS example from the dataset
generated_explanation = generate_explanation(css_example)
print("Generated Explanation for CSS Code:\n", generated_explanation)
