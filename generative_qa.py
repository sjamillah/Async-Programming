import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
import re
import numpy as np
from transformers import (
    TFAutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    create_optimizer
)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK punkt: {e}")

# Load evaluation metrics
try:
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')
except Exception as e:
    print(f"Warning: Could not load evaluation metrics: {e}")
    rouge = None
    bertscore = None

def clean_text(text):
    """Clean text by removing special characters and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_exact_match(prediction, reference):
    """Calculate if prediction exactly matches reference."""
    return 1.0 if clean_text(prediction) == clean_text(reference) else 0.0

def calculate_f1(prediction, reference):
    """Calculate F1 score between prediction and reference."""
    pred_tokens = set(word_tokenize(clean_text(prediction)))
    ref_tokens = set(word_tokenize(clean_text(reference)))
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    common = pred_tokens.intersection(ref_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_bleu(prediction, reference):
    """Calculate BLEU score between prediction and reference."""
    pred_tokens = word_tokenize(clean_text(prediction))
    ref_tokens = word_tokenize(clean_text(reference))
    
    # Use smoothing to handle short sequences
    smoothing = SmoothingFunction().method1
    try:
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        return score
    except:
        return 0.0

def evaluate_model(model, tokenizer, test_questions, test_answers, batch_size=8):
    """
    Evaluate the model's performance using multiple metrics.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer used for the model
        test_questions: List of test questions
        test_answers: List of reference answers
        batch_size: Size of batches for evaluation
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    if not isinstance(test_questions, list) or not isinstance(test_answers, list):
        raise ValueError("test_questions and test_answers must be lists")
    
    if len(test_questions) != len(test_answers):
        raise ValueError("test_questions and test_answers must have the same length")
    
    predictions = []
    references = []
    
    print("\nEvaluating model performance...")
    
    # Process in batches
    for i in range(0, len(test_questions), batch_size):
        batch_questions = test_questions[i:i + batch_size]
        batch_answers = test_answers[i:i + batch_size]
        
        try:
            # Prepare inputs
            inputs = tokenizer(
                batch_questions,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="tf"
            )
            
            # Get model predictions
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            # Decode predictions
            batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_predictions)
            references.extend(batch_answers)
            
        except Exception as e:
            print(f"Error in batch {i//batch_size}: {e}")
            predictions.extend([""] * len(batch_questions))
            references.extend(batch_answers)
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    
    # Exact Match
    exact_matches = [calculate_exact_match(p, r) for p, r in zip(predictions, references)]
    exact_match_score = np.mean(exact_matches)
    
    # F1 Score
    f1_scores = [calculate_f1(p, r) for p, r in zip(predictions, references)]
    avg_f1 = np.mean(f1_scores)
    
    # BLEU Score
    bleu_scores = [calculate_bleu(p, r) for p, r in zip(predictions, references)]
    avg_bleu = np.mean(bleu_scores)
    
    # ROUGE Score
    rouge_scores = {}
    if rouge is not None:
        try:
            rouge_scores = rouge.compute(predictions=predictions, references=references)
        except Exception as e:
            print(f"Warning: Could not calculate ROUGE scores: {e}")
            rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    # BERTScore
    bert_score = 0.0
    if bertscore is not None:
        try:
            bert_scores = bertscore.compute(
                predictions=predictions,
                references=references,
                model_type="microsoft/deberta-xlarge-mnli",
                batch_size=8
            )
            bert_score = np.mean(bert_scores['f1'])
        except Exception as e:
            print(f"Warning: Could not calculate BERTScore: {e}")
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Number of examples evaluated: {len(predictions)}")
    print(f"\nExact Match Score: {exact_match_score:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    if bertscore is not None:
        print(f"Average BERTScore: {bert_score:.4f}")
    if rouge is not None:
        print("\nROUGE Scores:")
        for key, value in rouge_scores.items():
            print(f"{key}: {value:.4f}")
    
    # Print example predictions
    print("\n=== Example Predictions ===")
    for i in range(min(5, len(test_questions))):
        print(f"\nQuestion: {test_questions[i]}")
        print(f"Reference Answer: {test_answers[i]}")
        print(f"Model Prediction: {predictions[i]}")
        print(f"Exact Match: {'✓' if exact_matches[i] == 1.0 else '✗'}")
        print(f"F1 Score: {f1_scores[i]:.4f}")
        print(f"BLEU Score: {bleu_scores[i]:.4f}")
        if bertscore is not None:
            print(f"BERTScore: {bert_scores['f1'][i]:.4f}")
        print("-" * 80)
    
    return {
        'exact_match': exact_match_score,
        'f1': avg_f1,
        'bleu': avg_bleu,
        'rouge': rouge_scores,
        'bert_score': bert_score,
        'predictions': predictions
    }

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    
    Args:
        file_path: Path to the CSV file containing the dataset
    
    Returns:
        Preprocessed training and validation datasets
    """
    print("\nLoading and preprocessing data...")
    
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Basic preprocessing
        df = df.dropna(subset=['Question', 'Answer'])
        df = df[df['Question'].str.len() > 0]
        df = df[df['Answer'].str.len() > 0]
        
        # Split into train and validation sets
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        return train_df, val_df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def prepare_dataset(df, tokenizer, max_length=512):
    """
    Prepare dataset for model training.
    
    Args:
        df: DataFrame containing questions and answers
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
    
    Returns:
        TensorFlow dataset ready for training
    """
    print("\nPreparing dataset...")
    
    try:
        # Prepare inputs
        inputs = ["question: " + q for q in df['Question'].tolist()]
        targets = df['Answer'].tolist()
        
        # Tokenize
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="tf"
        )
        
        labels = tokenizer(
            targets,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="tf"
        )
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': model_inputs['input_ids'],
                'attention_mask': model_inputs['attention_mask'],
                'labels': labels['input_ids']
            }
        ))
        
        return dataset
        
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        raise

def train_model(model, tokenizer, train_df, val_df, batch_size=8, epochs=10):
    """
    Train the model on the dataset.
    
    Args:
        model: Pre-trained model to fine-tune
        tokenizer: Tokenizer for the model
        train_df: Training data
        val_df: Validation data
        batch_size: Batch size for training
        epochs: Number of training epochs
    
    Returns:
        Trained model and training history
    """
    print("\nPreparing for training...")
    
    try:
        # Prepare datasets
        train_dataset = prepare_dataset(train_df, tokenizer)
        val_dataset = prepare_dataset(val_df, tokenizer)
        
        # Configure datasets
        train_dataset = train_dataset.shuffle(1000).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        
        # Create optimizer with learning rate schedule
        num_train_steps = len(train_dataset) * epochs
        optimizer, lr_schedule = create_optimizer(
            init_lr=2e-5,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_train_steps // 10
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                save_format='tf',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("\nStarting training...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

def main():
    try:
        # Load model and tokenizer
        model_name = "t5-base"
        print(f"\nLoading model and tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Load and preprocess data
        train_df, val_df = load_and_preprocess_data('medicalQ&A.csv')
        
        # Train model
        trained_model, history = train_model(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            val_df=val_df,
            batch_size=8,
            epochs=10
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        results = evaluate_model(
            model=trained_model,
            tokenizer=tokenizer,
            test_questions=val_df['Question'].tolist(),
            test_answers=val_df['Answer'].tolist()
        )
        
        # Save the final model
        print("\nSaving final model...")
        trained_model.save_pretrained('final_model')
        tokenizer.save_pretrained('final_model')
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
