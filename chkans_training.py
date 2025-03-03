from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load the pre-trained WangchanBERTa model and tokenizer
model_name = "model/wangchanberta"

def finetune_bert(model_name, 
                  save_path, 
                  bert_lr=2e-5, 
                  bert_tr_batchsize=8, 
                  bert_ev_batchsize=8, 
                  train_epoch=10, 
                  w_decay=0.01
                  ):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, load_in_8bit=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, local_files_only=True)  # binary classification

    # Prepare your dataset
    # Replace with your dataset loading and preprocessing
    dataset = load_dataset("your_dataset")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["question"], examples["input_answer"], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Set up the Trainer
    training_args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="epoch",
        learning_rate=bert_lr,
        per_device_train_batch_size=bert_tr_batchsize,
        per_device_eval_batch_size=bert_ev_batchsize,
        num_train_epochs=train_epoch,
        weight_decay=w_decay,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()
