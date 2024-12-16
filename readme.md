---

# Fine-Tuned DistilGPT-2

This repository contains code for fine-tuning and testing a `DistilGPT-2` model using the Alpaca dataset. The two key scripts provided in this repository are:

1. **fine_tuning.py** – Script for fine-tuning the `DistilGPT-2` model on the Alpaca dataset.
2. **Testing.py** – Script for testing the fine-tuned model and generating text based on given prompts.

## Project Overview

The goal of this project is to fine-tune the pre-trained `DistilGPT-2` model on the Alpaca dataset to generate responses for instructional prompts. Additionally, the repository includes code for testing the model’s performance by providing custom input prompts and receiving generated responses.

### Key Features

- Fine-tuning `DistilGPT-2` using the Alpaca dataset for instruction-response text generation.
- Efficient fine-tuning using **Low-Rank Adaptation (LoRA)**.
- Tokenizer and data preprocessing pipeline tailored for instructional text.
- Evaluation using custom prompts in the `Testing.py` script.

---

## File Descriptions

### `fine_tuning.py`

This script is used to fine-tune the `DistilGPT-2` model on the Alpaca dataset. It handles the following tasks:
- Loads the `DistilGPT-2` model and tokenizer.
- Prepares the Alpaca dataset for fine-tuning.
- Fine-tunes the model using Hugging Face's `Trainer` API.
- Saves the fine-tuned model in the `./fine_tuned_model` directory.

### `Testing.py`

Once the model is fine-tuned, use this script to test it:
- Loads the fine-tuned model and tokenizer from `./fine_tuned_model`.
- Accepts custom input prompts to generate responses.
- Displays the generated text based on the input.

---

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/gautamraj8044/fine-tuned-distilgpt2.git
cd fine-tuned-distilgpt2
```

### 2. Install Dependencies

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### 3. Fine-tune the Model

To start fine-tuning the model, run the following:

```bash
python fine_tuning.py
```

This will fine-tune the `DistilGPT-2` model on the Alpaca dataset and save the model in the `./fine_tuned_model` directory.

### 4. Test the Fine-Tuned Model

After fine-tuning, you can test the model using:

```bash
python Testing.py
```

You will be prompted to input a text, and the model will generate a response based on the fine-tuned parameters.

---

## Example Usage

- **Fine-tuning**: The model is trained on the Alpaca dataset consisting of instructional prompts and responses. The fine-tuned model can be used for generating instructional text.
  
- **Testing**: After fine-tuning, use the `Testing.py` script to input queries, and the model will generate relevant responses based on the instructions.

---

## License

This project is licensed under the MIT License.

---

## Contributing

Feel free to fork the repository, submit issues, or contribute to improving the model or adding new features via pull requests.

---
