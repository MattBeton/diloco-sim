import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from multiprocessing import cpu_count
from tqdm import tqdm


# Load the OpenWebText dataset from HuggingFace
def download_openwebtext():
    dataset = load_dataset("openwebtext", split="train[:20%]")
    return dataset


# Tokenize the dataset and append the EOT token
def tokenize_dataset_with_eot(dataset, tokenizer, eot_token_id):
    def tokenize_function(example):
        tokenized = tokenizer(example["text"], truncation=True, padding=False)
        # Append EOT token ID to the end of each tokenized input
        tokenized["input_ids"].append(eot_token_id)
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=False, num_proc=cpu_count())
    return tokenized_dataset


# Save the tokenized dataset into a .bin file
def save_tokenized_to_bin(tokenized_dataset, output_file, batch_size=1000):
    with open(output_file, "wb") as f:
        buffer = []
        for idx, entry in enumerate(tqdm(tokenized_dataset, desc="Saving to .bin file", unit="entries")):
            input_ids = entry["input_ids"]
            buffer.extend(input_ids)  # Collect all input_ids in a buffer

            # Write to file in batches
            if (idx + 1) % batch_size == 0:
                input_ids_np = np.array(buffer, dtype=np.uint16)
                input_ids_np.tofile(f)
                buffer = []  # Reset the buffer

        # Write remaining entries
        if buffer:
            input_ids_np = np.array(buffer, dtype=np.uint16)
            input_ids_np.tofile(f)


def main():
    print("Preparing OpenWebText dataset for training...")

    print("Step 1: Downloading OpenWebText dataset...")
    dataset = download_openwebtext()

    print(dataset)

    print("Step 2: Tokenizing the dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Using GPT-2 tokenizer

    eot_token_id = tokenizer.eos_token_id  # Typically 50256 for GPT-2

    tokenized_dataset = tokenize_dataset_with_eot(dataset, tokenizer, eot_token_id)

    print("Step 3: Saving the tokenized dataset to a .bin file...")
    output_file = "openwebtext.bin"
    save_tokenized_to_bin(tokenized_dataset, output_file)
    print(f"Tokenized dataset with EOT saved to {output_file}")


if __name__ == "__main__":
    main()

