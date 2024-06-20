import transformers
import numpy as np

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load the SQuAD dataset
dataset = load_dataset("squad")

# Select a smaller subset of examples (e.g., first 10000 examples)
small_dataset = dataset["train"].select(range(10000))
small_eval_dataset = dataset["validation"].select(range(5000))

# Define preprocess function
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)

        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if not (offset[context_start][0] <= start_char and offset[context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_position = context_start
            while start_position < len(offset) and offset[start_position][0] <= start_char:
                start_position += 1
            start_positions.append(start_position - 1)

            end_position = context_end
            while end_position >= 0 and offset[end_position][1] >= end_char:
                end_position -= 1
            end_positions.append(end_position + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_datasets = small_dataset.map(preprocess_function, batched=True)
tokenized_eval_datasets = small_eval_dataset.map(preprocess_function, batched=True)

# Split small dataset into train and eval datasets
train_dataset = tokenized_datasets
eval_dataset = tokenized_eval_datasets

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize Trainer with both train and eval datasets
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model fine-tuned and saved successfully.")