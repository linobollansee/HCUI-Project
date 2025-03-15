import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import torch
from huggingface_hub import login
import getpass  # Securely prompt for the API token

# 0. Prompt for Hugging Face API Token
hf_token = getpass.getpass("Enter your Hugging Face API token: ")
login(token=hf_token)

# 1. List all CSV files in the current directory
csv_files = sorted([f for f in os.listdir() if f.endswith(".csv")])

if not csv_files:
    raise FileNotFoundError("No CSV files found in the current directory.")

# 2. Prompt user to select a CSV file
selected_file = None

for file in csv_files:
    user_input = input(f"Use this file? {file} (yes/no): ").strip().lower()
    if user_input in ["yes", "y"]:
        selected_file = file
        break

if selected_file is None:
    raise ValueError("No CSV file selected. Exiting.")

print(f"Using file: {selected_file}")

# 3. Load the selected dataset
df = pd.read_csv(selected_file)

# Display available columns
print("Available columns in the dataset:", df.columns.tolist())

# 4. Detect possible text-based columns
text_columns = [col for col in df.columns if df[col].dtype == 'object']

if not text_columns:
    raise ValueError("No text-based columns found in the dataset. Ensure your CSV contains textual data.")

# 5. Ask if the user wants to use the detected columns
print("Detected text-based columns:", text_columns)
use_detected = input("Do you want to use these detected columns? (yes/no): ").strip().lower()

if use_detected in ["yes", "y"]:
    target_column = text_columns[0]  # Default to first detected column
else:
    # Allow manual selection
    print("Please select a column from the list:")
    for i, col in enumerate(text_columns):
        print(f"{i + 1}. {col}")
    choice = int(input("Enter the number corresponding to your chosen column: ")) - 1
    target_column = text_columns[choice]

print(f"Using column '{target_column}' for training.")

# Convert the DataFrame to a Hugging Face dataset
dataset = Dataset.from_pandas(df)

# 6. Load the tokenizer and pre-trained model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# 7. Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples[target_column], truncation=True, padding="max_length", max_length=512)

# Apply preprocessing to the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 8. Split dataset into train and test sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 9. Set up the data collator and training arguments
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)

# 10. Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 11. Fine-tune the model
trainer.train()

# 12. Save the trained model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# 13. Load the fine-tuned model and tokenizer for inference
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 14. Generate explanations for text samples
def generate_explanation(text_sample, max_length=150):
    inputs = tokenizer(text_sample, return_tensors="pt")
    
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
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 15. Example usage
text_example = df[target_column].iloc[0]
generated_explanation = generate_explanation(text_example)
print("Generated Explanation:\n", generated_explanation)
