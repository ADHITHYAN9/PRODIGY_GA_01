import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

# Step 1: Load Dataset (Create a Custom Dataset Class)
class CustomTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        self.examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Check if text is empty
        if not text:
            raise ValueError("The dataset file is empty.")
        
        # Tokenize text and create input sequences
        tokenized_text = tokenizer.encode(text, truncation=True, padding=False)
        
        # If the tokenized text is shorter than block_size, use the entire text as one example
        if len(tokenized_text) < block_size:
            self.examples.append(tokenized_text)
        else:
            # Split the tokenized text into blocks of size `block_size`
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(tokenized_text[i:i + block_size])
        
        # Check if examples were created
        if len(self.examples) == 0:
            raise ValueError(f"No examples created from the dataset with block size {block_size}.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

# Step 2: Load Pre-trained GPT-2 Model and Tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add special tokens if necessary
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is the same as eos token

# Step 3: Create Dataset
file_path = "D:/GEN AI PROJECT/dataset.txt"  # Replace with your dataset path
block_size = 128  # You can experiment with smaller block sizes like 64 or 32
try:
    dataset = CustomTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=block_size)
    print(f"Dataset created with {len(dataset)} examples.")
except ValueError as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Step 4: Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
)

# Step 5: Use Data Collator for Language Modeling (Handles Padding)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT-2 is causal language model (not masked)
)

# Step 6: Trainer for Fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,  # Include the data collator to handle padding
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save the Fine-tuned Model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

# Step 9: Generate Text with the Fine-tuned Model
def generate_text(prompt, max_length=50, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "Once upon a time"
print(generate_text(prompt, max_length=100))
