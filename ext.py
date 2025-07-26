!pip install tensorflow
!pip install transformers
!pip install datasets
!pip install -U nltk
!pip install -U spacy
!pip install evaluate
!python -m spacy download en_core_web_sm
!pip install rouge_score
!pip install scikit-learn

import tensorflow as tf
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    TFBertForQuestionAnswering,
    BertTokenizer,
    BertConfig
)
import nltk
import spacy
import evaluate
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Download required NLTK data
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Load dataset (replace with your actual dataset path)
ds = '/content/medicalQ&A.csv'  # Replace with your actual file path
def load_data(ds):
    try:
        df = pd.read_csv(ds)
        print(f"Dataset loaded successfully with {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: Could not find {ds}. Creating sample data for demonstration.")
        # Create sample data
        sample_data = {
            'Question': [
                'What are the symptoms of diabetes?',
                'How is hypertension treated?',
                'What causes heart disease?',
                'What are the side effects of aspirin?',
                'How to prevent stroke?'
            ] * 10,  # Only 50 rows for quick test
            'Answer': [
                'Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, and blurred vision.',
                'Hypertension is treated with lifestyle changes and medications.',
                'Heart disease is caused by high cholesterol, smoking, diabetes, and high blood pressure.',
                'Common side effects of aspirin include stomach irritation and bleeding.',
                'Stroke prevention includes controlling blood pressure and not smoking.'
            ] * 10
        }
        df = pd.DataFrame(sample_data)
    return df

df = load_data(ds)

# Filter columns
filtered_df = df[['Question', 'Answer']].copy()

# Clean text
import re
from nltk.corpus import stopwords
def clean_text(text, remove_stopwords=False):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?\.\,\!\-]', '', text)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    return text.strip()
filtered_df['Cleaned_Question'] = filtered_df['Question'].apply(lambda x: clean_text(x, remove_stopwords=False))
filtered_df['Cleaned_Answer'] = filtered_df['Answer'].apply(lambda x: clean_text(x, remove_stopwords=False))
filtered_df = filtered_df.dropna(subset=['Cleaned_Question', 'Cleaned_Answer'])
filtered_df = filtered_df[filtered_df['Cleaned_Question'].str.len() > 5]
filtered_df = filtered_df[filtered_df['Cleaned_Answer'].str.len() > 10]

print(f"After cleaning: {len(filtered_df)} samples")

# Data exploration and visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distribution of question lengths
question_lengths = filtered_df['Cleaned_Question'].str.len()
axes[0, 0].hist(question_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Question Lengths')
axes[0, 0].set_xlabel('Question Length (characters)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(question_lengths.mean(), color='red', linestyle='--', label=f'Mean: {question_lengths.mean():.1f}')
axes[0, 0].legend()

# 2. Distribution of answer lengths
answer_lengths = filtered_df['Cleaned_Answer'].str.len()
axes[0, 1].hist(answer_lengths, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Distribution of Answer Lengths')
axes[0, 1].set_xlabel('Answer Length (characters)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(answer_lengths.mean(), color='red', linestyle='--', label=f'Mean: {answer_lengths.mean():.1f}')
axes[0, 1].legend()

# 3. Question vs Answer length correlation
axes[1, 0].scatter(question_lengths, answer_lengths, alpha=0.6, color='coral')
axes[1, 0].set_title('Question vs Answer Length Correlation')
axes[1, 0].set_xlabel('Question Length (characters)')
axes[1, 0].set_ylabel('Answer Length (characters)')
correlation = np.corrcoef(question_lengths, answer_lengths)[0, 1]
axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=axes[1, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 4. Word count distributions
question_word_counts = filtered_df['Cleaned_Question'].str.split().str.len()
answer_word_counts = filtered_df['Cleaned_Answer'].str.split().str.len()

axes[1, 1].hist(question_word_counts, bins=20, alpha=0.7, label='Questions', color='lightblue')
axes[1, 1].hist(answer_word_counts, bins=20, alpha=0.7, label='Answers', color='lightcoral')
axes[1, 1].set_title('Word Count Distribution')
axes[1, 1].set_xlabel('Word Count')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Split data
X = filtered_df['Cleaned_Question']
y = filtered_df['Cleaned_Answer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

# Further split training data for validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train_final)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Use a smaller model for speed if needed
model_name = 'bert-base-uncased'  # Use distilbert for faster training
print(f"Loading BERT model: {model_name}")
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
model = TFBertForQuestionAnswering.from_pretrained(model_name, config=config)

# Data preparation for extractive QA
import numpy as np
def prepare_bert_qa_data(questions, contexts, max_length=384):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    start_positions = []
    end_positions = []
    for question, context in zip(questions, contexts):
        encoded = tokenizer(
            question,
            context,
            max_length=max_length,
            padding='max_length',
            truncation='only_second',
            return_tensors='np',
            return_token_type_ids=True
        )
        # Place answer at the start of context for demo
        start_pos = encoded['input_ids'][0].tolist().index(tokenizer.cls_token_id) + 1
        end_pos = start_pos + min(10, max_length - start_pos - 1)
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])
        token_type_ids.append(encoded['token_type_ids'][0])
        start_positions.append(start_pos)
        end_positions.append(end_pos)
    return {
        'input_ids': np.array(input_ids),
        'attention_mask': np.array(attention_masks),
        'token_type_ids': np.array(token_type_ids),
        'start_positions': np.array(start_positions),
        'end_positions': np.array(end_positions)
    }

train_data = prepare_bert_qa_data(X_train_final.tolist(), y_train_final.tolist())
val_data = prepare_bert_qa_data(X_val.tolist(), y_val.tolist())
test_data = prepare_bert_qa_data(X_test.tolist(), y_test.tolist())

def create_bert_tf_dataset(data, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices({
        'input_ids': data['input_ids'],
        'attention_mask': data['attention_mask'],
        'token_type_ids': data['token_type_ids'],
        'start_positions': data['start_positions'],
        'end_positions': data['end_positions']
    })
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_dataset = create_bert_tf_dataset(train_data, batch_size=4)
val_dataset = create_bert_tf_dataset(val_data, batch_size=4)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
model.compile(
    optimizer=optimizer,
    loss={
        'start_logits': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'end_logits': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    },
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=1,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1
    )
]

print("Starting BERT model training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=2,  # Only 2 epochs for speed
    callbacks=callbacks,
    verbose=1
)

print("Training complete. Evaluating model...")
# Evaluation code can be added here as needed

# Comprehensive evaluation functions for BERT
def extract_bert_answer_from_logits(input_ids, start_logits, end_logits, tokenizer):
    """Extract answer from BERT model logits"""
    start_idx = tf.argmax(start_logits, axis=-1).numpy()
    end_idx = tf.argmax(end_logits, axis=-1).numpy()

    # Ensure end_idx >= start_idx
    if end_idx < start_idx:
        end_idx = start_idx

    # Ensure indices are within bounds
    start_idx = max(0, min(start_idx, len(input_ids) - 1))
    end_idx = max(start_idx, min(end_idx, len(input_ids) - 1))

    # Extract answer tokens
    answer_tokens = input_ids[start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer.strip()

def calculate_exact_match(predicted, actual):
    """Calculate exact match score"""
    return 1 if predicted.lower().strip() == actual.lower().strip() else 0

def calculate_token_f1(predicted, actual):
    """Calculate F1 score based on token overlap"""
    pred_tokens = set(predicted.lower().split())
    actual_tokens = set(actual.lower().split())

    if not pred_tokens and not actual_tokens:
        return 1.0
    if not pred_tokens or not actual_tokens:
        return 0.0

    common = pred_tokens.intersection(actual_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(actual_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1

# Load evaluation metrics
rouge_evaluator = evaluate.load("rouge")
bleu_evaluator = evaluate.load("bleu")

def comprehensive_bert_evaluation(model, test_questions, test_answers, tokenizer, batch_size=4):
    """Comprehensive evaluation with multiple metrics for BERT"""
    predictions = []
    actual_answers = []

    # Prepare test data
    test_data = prepare_bert_qa_data(test_questions, test_answers)
    test_dataset = create_bert_tf_dataset(test_data, batch_size)

    print("Generating predictions with BERT...")

    # Get predictions
    batch_idx = 0
    for batch in test_dataset:
        try:
            outputs = model.predict(batch, verbose=0)

            batch_size_actual = len(batch['input_ids'])

            for i in range(batch_size_actual):
                # Extract answer from logits
                predicted_answer = extract_bert_answer_from_logits(
                    batch['input_ids'][i].numpy(),
                    outputs['start_logits'][i],
                    outputs['end_logits'][i],
                    tokenizer
                )

                predictions.append(predicted_answer)
                actual_idx = batch_idx * batch_size + i
                if actual_idx < len(test_answers):
                    actual_answers.append(test_answers[actual_idx])

            batch_idx += 1

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    # Ensure we have the right number of predictions
    predictions = predictions[:len(test_answers)]
    actual_answers = test_answers[:len(predictions)]

    # Calculate metrics
    exact_matches = [calculate_exact_match(pred, actual) for pred, actual in zip(predictions, actual_answers)]
    token_f1_scores = [calculate_token_f1(pred, actual) for pred, actual in zip(predictions, actual_answers)]

    # Calculate aggregate metrics
    exact_match_score = np.mean(exact_matches)
    avg_token_f1 = np.mean(token_f1_scores)

    # ROUGE scores
    rouge_results = rouge_evaluator.compute(predictions=predictions, references=actual_answers)

    # BLEU scores
    bleu_results = bleu_evaluator.compute(predictions=predictions, references=[[ref] for ref in actual_answers])

    metrics = {
        'exact_match': exact_match_score,
        'token_f1': avg_token_f1,
        'rouge1': rouge_results['rouge1'],
        'rouge2': rouge_results['rouge2'],
        'rougeL': rouge_results['rougeL'],
        'bleu': bleu_results['bleu']
    }

    return predictions, actual_answers, metrics

# Evaluate the BERT model
print("Evaluating BERT model...")
predictions, actual_answers, metrics = comprehensive_bert_evaluation(
    model, X_test.tolist(), y_test.tolist(), tokenizer
)

# Print results
print("\n" + "="*60)
print("    BERT COMPREHENSIVE EVALUATION RESULTS")
print("="*60)
print(f"Model: {model_name}")
print(f"Total test samples: {len(predictions)}")
print(f"Exact Match Score: {metrics['exact_match']:.4f}")
print(f"Token F1 Score: {metrics['token_f1']:.4f}")
print(f"ROUGE-1: {metrics['rouge1']:.4f}")
print(f"ROUGE-2: {metrics['rouge2']:.4f}")
print(f"ROUGE-L: {metrics['rougeL']:.4f}")
print(f"BLEU Score: {metrics['bleu']:.4f}")
print("="*60)

# Sample predictions
print("\nSample BERT Predictions:")
print("-" * 60)
for i in range(min(5, len(predictions))):
    print(f"\nQuestion: {X_test.iloc[i]}")
    print(f"Actual Answer: {actual_answers[i][:100]}...")
    print(f"Predicted Answer: {predictions[i][:100]}...")
    print(f"Token F1: {calculate_token_f1(predictions[i], actual_answers[i]):.3f}")
    print("-" * 60)

# Metrics visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Metrics comparison
metric_names = ['Exact Match', 'Token F1', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']
metric_values = [
    metrics['exact_match'], metrics['token_f1'], metrics['rouge1'],
    metrics['rouge2'], metrics['rougeL'], metrics['bleu']
]

bars = axes[0, 0].bar(metric_names, metric_values, color='skyblue', alpha=0.8)
axes[0, 0].set_title('BERT Model Performance Metrics')
axes[0, 0].set_ylabel('Score')
axes[0, 0].tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, metric_values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

# 2. Token F1 score distribution
token_f1_scores = [calculate_token_f1(pred, actual) for pred, actual in zip(predictions, actual_answers)]
axes[0, 1].hist(token_f1_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Distribution of Token F1 Scores')
axes[0, 1].set_xlabel('Token F1 Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(np.mean(token_f1_scores), color='red', linestyle='--',
                   label=f'Mean: {np.mean(token_f1_scores):.3f}')
axes[0, 1].legend()

# 3. Answer length comparison
pred_lengths = [len(pred.split()) for pred in predictions]
actual_lengths = [len(actual.split()) for actual in actual_answers]

axes[1, 0].scatter(actual_lengths, pred_lengths, alpha=0.6, color='coral')
axes[1, 0].plot([0, max(max(pred_lengths), max(actual_lengths))],
                [0, max(max(pred_lengths), max(actual_lengths))], 'r--', alpha=0.8)
axes[1, 0].set_title('Predicted vs Actual Answer Lengths')
axes[1, 0].set_xlabel('Actual Answer Length (words)')
axes[1, 0].set_ylabel('Predicted Answer Length (words)')

# 4. Performance by answer length
length_buckets = defaultdict(list)
for i, (pred, actual) in enumerate(zip(predictions, actual_answers)):
    actual_len = len(actual.split())
    bucket = f"{(actual_len // 5) * 5}-{(actual_len // 5) * 5 + 4}"
    length_buckets[bucket].append(token_f1_scores[i])

bucket_names = sorted(length_buckets.keys(), key=lambda x: int(x.split('-')[0]))
bucket_means = [np.mean(length_buckets[bucket]) for bucket in bucket_names]

axes[1, 1].bar(range(len(bucket_names)), bucket_means, color='lightcoral', alpha=0.8)
axes[1, 1].set_title('Performance by Answer Length')
axes[1, 1].set_xlabel('Answer Length Range (words)')
axes[1, 1].set_ylabel('Average Token F1 Score')
axes[1, 1].set_xticks(range(len(bucket_names)))
axes[1, 1].set_xticklabels(bucket_names, rotation=45)

plt.tight_layout()
plt.show()

# Save the model
print("\nSaving BERT model...")
model.save_pretrained('./bert_medical_qa_model')
tokenizer.save_pretrained('./bert_medical_qa_model')

print("\nBERT model training and evaluation completed!")
print("The BERT model has been trained for extractive question answering with comprehensive evaluation metrics.")
print("Model and tokenizer saved to './bert_medical_qa_model/'")

# Function to make predictions with the trained BERT model
def predict_answer(question, context, model, tokenizer, max_length=512):
    """Make a prediction using the trained BERT model"""
    # Encode the question and context
    encoded = tokenizer(
        question,
        context,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    # Get model predictions
    outputs = model(encoded)

    # Extract answer
    start_idx = tf.argmax(outputs.start_logits, axis=-1).numpy()[0]
    end_idx = tf.argmax(outputs.end_logits, axis=-1).numpy()[0]

    # Ensure valid indices
    if end_idx < start_idx:
        end_idx = start_idx

    # Extract answer tokens
    input_ids = encoded['input_ids'][0].numpy()
    answer_tokens = input_ids[start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer.strip()

# Example usage
print("\nExample predictions:")
sample_questions = [
    "What are the symptoms of diabetes?",
    "How is hypertension treated?"
]

sample_contexts = [
    "Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, and blurred vision. Other symptoms may include slow-healing wounds and frequent infections.",
    "Hypertension is treated with lifestyle changes like diet and exercise, and medications such as ACE inhibitors, beta-blockers, and diuretics."
]

for i, (question, context) in enumerate(zip(sample_questions, sample_contexts)):
    predicted_answer = predict_answer(question, context, model, tokenizer)
    print(f"\nExample {i+1}:")
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Predicted Answer: {predicted_answer}")