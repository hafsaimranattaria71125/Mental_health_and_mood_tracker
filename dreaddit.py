# -*- coding: utf-8 -*-


# ============================================================================
# MENTAL HEALTH MOOD TRACKER - MODEL 1: DREADDIT STRESS DETECTION
# Fine-tune MentalRoBERTa for Binary Stress Classification
# ============================================================================

# CELL 1: Install Dependencies
# ============================================================================

!pip install transformers datasets torch scikit-learn pandas numpy
!pip install accelerate scipy

print("✅ Dependencies installed successfully!")

# ============================================================================
# CELL 2: Import Libraries
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from datasets import Dataset
# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# CELL 3: Upload and Load DREADDIT Dataset
# ============================================================================

from google.colab import files

print("📁 Step 1: Upload your Training CSV file (with 'text' and 'label' columns)")
uploaded_train = files.upload()
train_filename = list(uploaded_train.keys())[0]
train_df_raw = pd.read_csv(train_filename)
print(f"✅ Uploaded Training file: {train_filename} | Shape: {train_df_raw.shape}")

print("\n📁 Step 2: Upload your Testing CSV file (with 'text' and 'label' columns)")
uploaded_test = files.upload()
test_filename = list(uploaded_test.keys())[0]
test_df_raw = pd.read_csv(test_filename)
print(f"✅ Uploaded Testing file: {test_filename} | Shape: {test_df_raw.shape}")

# ============================================================================
# CELL 4: Explore DREADDIT Dataset
# ============================================================================
def explore_dataset(df):
  print("\n" + "="*60)
  print("DATASET EXPLORATION")
  print("="*60)

  # Check for missing values
  print(f"\nMissing values:")
  print(df.isnull().sum())

  # Check data types
  print(f"\nData types:")
  print(df.dtypes)

  # Label distribution
  print(f"\n Label Distribution:")
  label_dist = df['label'].value_counts()
  print(label_dist)
  print(f"\nPercentage:")
  print(df['label'].value_counts(normalize=True) * 100)

# Text length statistics
  df['text_length'] = df['text'].str.len()
  print(f"\n📏 Text Length Statistics:")
  print(df['text_length'].describe())

# Visualization
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Label distribution pie chart
  label_dist.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
  axes[0].set_title('Stress Label Distribution')
  axes[0].set_ylabel('Count')
  axes[0].set_xlabel('Label')

# Text length histogram
  axes[1].hist(df['text_length'], bins=50, color='#3498db', edgecolor='black')
  axes[1].set_title('Text Length Distribution')
  axes[1].set_xlabel('Character Count')
  axes[1].set_ylabel('Frequency')

  plt.tight_layout()
  plt.show()

print(f"\n✅ Train Dataset exploration !")
explore_dataset(train_df_raw)
print(f"\n✅ Test Dataset exploration !")
explore_dataset(test_df_raw)

# ============================================================================
# CELL 5: Data Preprocessing & Validation Split
# ============================================================================

print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

def clean_dataframe(df, name="Dataset"):
    # Remove NaN values
    df_clean = df.dropna(subset=['text', 'label']).copy()
    # Remove duplicates based on text
    df_clean = df_clean.drop_duplicates(subset=['text'])
    # Remove very short texts
    df_clean['text'] = df_clean['text'].astype(str).str.strip()
    df_clean = df_clean[df_clean['text'].str.len() >= 25]

    # Ensure labels are strictly integers (0 or 1)
    df_clean['label'] = df_clean['label'].astype(int)

    print(f"✅ Cleaned {name}. Rows remaining: {len(df_clean)}")
    return df_clean

# Clean both sets independently
print("Processing Train Set...")
train_df_clean = clean_dataframe(train_df_raw, "Train Set")

print("\nProcessing Test Set...")
test_df = clean_dataframe(test_df_raw, "Test Set")

# Create a validation set from the training pool (e.g., 15% for evaluation)
train_df, val_df = train_test_split(
    train_df_clean,
    test_size=0.15,
    stratify=train_df_clean['label'],
    random_state=42
)

print(f"\n📊 Final Partition Breakdown:")
print(f"   Train set:      {len(train_df)} samples ({len(train_df)/len(train_df_clean)*100:.1f}% of train file)")
print(f"   Validation set: {len(val_df)} samples ({len(val_df)/len(train_df_clean)*100:.1f}% of train file)")
print(f"   Test set:       {len(test_df)} samples (fully independent)")

print(f"\n🏷️  Train Label distribution:\n{train_df['label'].value_counts()}")
print(f"\n🏷️  Test Label distribution:\n{test_df['label'].value_counts()}")

# ============================================================================
# CELL 6: Load Model and Tokenizer
# ============================================================================
from google.colab import userdata
from huggingface_hub import login
print("\n" + "="*60)
print("LOADING MODEL")
print("="*60)

# Use MentalRoBERTa base model

model_name = "mental/mental-roberta-base"


login(userdata.get('HF_TOKEN'))
print(f"\n📥 Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # Binary classification (stressed/not stressed)
    id2label={0: "Not Stressed", 1: "Stressed"},
    label2id={"Not Stressed": 0, "Stressed": 1}
)

# Move to GPU if available
model.to(device)

print(f"✅ Model and tokenizer loaded successfully!")
print(f"   Model parameters: {model.num_parameters():,}")

# ============================================================================
# CELL 7: Tokenize Datasets
# ============================================================================

print("\n" + "="*60)
print("TOKENIZING DATASETS")
print("="*60)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({
    "text": train_df["text"].tolist(),
    "label": train_df["label"].tolist()
})

val_dataset = Dataset.from_dict({
    "text": val_df["text"].tolist(),
    "label": val_df["label"].tolist()
})

test_dataset = Dataset.from_dict({
    "text": test_df["text"].tolist(),
    "label": test_df["label"].tolist()
})

print(f"\n🔄 Tokenizing train dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=32)
train_dataset = train_dataset.remove_columns(["text"])

print(f"🔄 Tokenizing validation dataset...")
val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=32)
val_dataset = val_dataset.remove_columns(["text"])

print(f"🔄 Tokenizing test dataset...")
test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=32)
test_dataset = test_dataset.remove_columns(["text"])

print(f"✅ Tokenization complete!")

# ============================================================================
# CELL 8: Define Metrics Function
# ============================================================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

print("✅ Metrics function defined!")

# ============================================================================
# CELL 9: Training Arguments
# ============================================================================

print("\n" + "="*60)
print("TRAINING SETUP")
print("="*60)

training_args = TrainingArguments(
    output_dir="./dreaddit_stress_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    report_to="tensorboard",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    learning_rate=2e-5,
    seed=42,
    gradient_accumulation_steps=1
)

print(f"✅ Training arguments set:")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Learning rate: {training_args.learning_rate}")

# ============================================================================
# CELL 10: Create Trainer and Train
# ============================================================================

print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print(f"\n🚀 Starting training...")
print(f"⏱️  This will take approximately 5-15 minutes on GPU...\n")

train_result = trainer.train()

print(f"\n✅ Training completed!")
print(f"\n📊 Training Results:")
for key, value in train_result.metrics.items():
    print(f"   {key}: {value:.4f}")

# ============================================================================
# CELL 11: Evaluate on Validation Set
# ============================================================================

print("\n" + "="*60)
print("VALIDATION EVALUATION")
print("="*60)

print(f"\n🔍 Evaluating on validation set...")
eval_result = trainer.evaluate()

print(f"\n✅ Validation Results:")
for key, value in eval_result.items():
    print(f"   {key}: {value:.4f}")

# ============================================================================
# CELL 12: Evaluate on Test Set
# ============================================================================

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

print(f"\n🔍 Evaluating on test set...")
predictions_output = trainer.predict(test_dataset)
predictions = np.argmax(predictions_output.predictions, axis=1)

# Get true labels
test_labels = test_dataset["label"]

# Calculate metrics
test_accuracy = accuracy_score(test_labels, predictions)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    test_labels,
    predictions,
    average="weighted",
    zero_division=0
)

print(f"\n✅ Test Set Results:")
print(f"   Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {test_f1:.4f}")

# Detailed classification report
print(f"\n📋 Detailed Classification Report:")
print(classification_report(
    test_labels,
    predictions,
    target_names=["Not Stressed", "Stressed"],
    digits=4
))

# Confusion matrix
cm = confusion_matrix(test_labels, predictions)
print(f"\n🔲 Confusion Matrix:")
print(cm)

# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Stressed', 'Stressed'],
    yticklabels=['Not Stressed', 'Stressed'],
    ax=ax,
    cbar=True
)
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
ax.set_title('DREADDIT Stress Detection - Confusion Matrix')
plt.tight_layout()
plt.show()


# ============================================================================
# CELL 14: Save Model and Tokenizer
# ============================================================================

print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

output_dir = "./dreaddit_stress_model_final"

print(f"\n💾 Saving model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model and tokenizer saved!")

# Save training results to JSON
results_dict = {
    "model": model_name,
    "task": "Binary Stress Detection (DREADDIT)",
    "test_accuracy": float(test_accuracy),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),
    "test_samples": len(test_labels),
    "confusion_matrix": cm.tolist(),
    "timestamp": datetime.now().isoformat()
}

with open(f"{output_dir}/results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print(f"✅ Results saved to results.json")
