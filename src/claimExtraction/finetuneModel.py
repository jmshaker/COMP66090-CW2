import wandb
import torch
import gc
import json
import random
import numpy as np
#import sys
#sys.path.append('.')

from transformers import BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW, DistilBertForSequenceClassification, RobertaForSequenceClassification
from datasets import load_dataset, load_metric, Features, ClassLabel

labels = ClassLabel(num_classes=2, names=['NOTCLAIM', 'CLAIM'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(predictions)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    str_2_int = { "NOTCLAIM": 0, "CLAIM": 1 }
    tokenized_batch = tokenizer(examples['sentence'], padding="max_length", truncation=True)        
    tokenized_batch['label'] = [str_2_int[label] for label in examples['label']]  
    return tokenized_batch


#wandb.init() #6373b880383af25c4fd46232924ec59a0cf9456c  
    
metric = load_metric("accuracy")

tokenizer = AutoTokenizer.from_pretrained("roberta-large")

model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2)

dataset = load_dataset('json', data_files={
    'train': 'BERT/data/train.json',
    'test': 'BERT/data/test.json'
})

train_dataset = dataset['train']
test_dataset = dataset['test']

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(['sentence_text', 'tokens', 'pos'])
test_dataset = test_dataset.remove_columns(['sentence_text', 'tokens', 'pos'])

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

training_args = TrainingArguments(
    output_dir='models/roberta-large/',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='logs',            # directory for storing logs
    save_steps=1000,
    logging_steps=100,
    learning_rate=5e-6,
    do_eval=True
)

gc.collect()
torch.cuda.empty_cache()

trainer = Trainer(
    model=model, train_dataset=train_dataset, args=training_args, eval_dataset=test_dataset, compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()