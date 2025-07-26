import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import (
    TFAutoModelForQuestionAnswering,
    AutoTokenizer,
    create_optimizer
)
from sklearn.model_selection import train_test_split
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords')
    nltk.download('punkt')
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

# Load evaluation metrics
try:
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')
except Exception as e:
    print(f"Warning: Could not load evaluation metrics: {e}")
    rouge = None
    bertscore = None

class ExtractiveQA:
    def __init__(self, model_name="bert-base-uncased", max_length=512):
        """
        Initialize the Extractive QA model.
        
        Args:
            model_name: Name of the pre-trained model to use
            max_length: Maximum sequence length for input
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
        
    def preprocess_text(self, text):
        """
        Clean and preprocess text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep question marks
        text = re.sub(r'[^\w\s\?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def prepare_inputs(self, questions, contexts):
        """
        Prepare inputs for the model.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            
        Returns:
            Dictionary containing input_ids and attention_mask
        """
        # Tokenize inputs
        encodings = self.tokenizer(
            questions,
            contexts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="tf"
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
    
    def find_answer_span(self, context, answer, question):
        """
        Find the start and end positions of the answer in the context.
        
        Args:
            context: The context text
            answer: The answer text
            question: The question text
            
        Returns:
            Tuple of (start_position, end_position) or (None, None) if not found
        """
        # Tokenize the full input
        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="tf"
        )
        
        # Get the input IDs
        input_ids = encoding['input_ids'][0]
        
        # Find the start of the context
        question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        context_start = len(question_tokens) + 2  # +2 for [CLS] and [SEP]
        
        # Tokenize the answer
        answer_tokens = self.tokenizer.encode(
            answer,
            add_special_tokens=False,
            max_length=self.max_length - context_start,
            truncation=True
        )
        
        # Find the answer span in the context
        context_tokens = self.tokenizer.encode(
            context,
            add_special_tokens=False,
            max_length=self.max_length - context_start,
            truncation=True
        )
        
        # Search for the answer span
        for i in range(len(context_tokens) - len(answer_tokens) + 1):
            if context_tokens[i:i + len(answer_tokens)] == answer_tokens:
                start_pos = i + context_start
                end_pos = i + len(answer_tokens) - 1 + context_start
                if end_pos < self.max_length:
                    return start_pos, end_pos
        
        return None, None
    
    def prepare_dataset(self, questions, contexts, answers):
        """
        Prepare dataset for training.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            answers: List of answers
            
        Returns:
            TensorFlow dataset ready for training
        """
        input_ids = []
        attention_mask = []
        start_positions = []
        end_positions = []
        
        for question, context, answer in zip(questions, contexts, answers):
            # Find answer span
            start_pos, end_pos = self.find_answer_span(context, answer, question)
            
            if start_pos is None or end_pos is None:
                continue
            
            # Prepare inputs
            encoding = self.tokenizer(
                question,
                context,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="tf"
            )
            
            input_ids.append(encoding['input_ids'][0])
            attention_mask.append(encoding['attention_mask'][0])
            start_positions.append(start_pos)
            end_positions.append(end_pos)
        
        # Convert to tensors
        input_ids = tf.stack(input_ids)
        attention_mask = tf.stack(attention_mask)
        start_positions = tf.convert_to_tensor(start_positions, dtype=tf.int32)
        end_positions = tf.convert_to_tensor(end_positions, dtype=tf.int32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'start_positions': start_positions,
                'end_positions': end_positions
            }
        ))
        
        return dataset
    
    def train(self, train_questions, train_contexts, train_answers,
              val_questions=None, val_contexts=None, val_answers=None,
              batch_size=8, epochs=3, learning_rate=2e-5):
        """
        Train the model.
        
        Args:
            train_questions: List of training questions
            train_contexts: List of training contexts
            train_answers: List of training answers
            val_questions: List of validation questions (optional)
            val_contexts: List of validation contexts (optional)
            val_answers: List of validation answers (optional)
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Training history
        """
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_questions, train_contexts, train_answers)
        train_dataset = train_dataset.shuffle(1000).batch(batch_size)
        
        if val_questions is not None:
            val_dataset = self.prepare_dataset(val_questions, val_contexts, val_answers)
            val_dataset = val_dataset.batch(batch_size)
        else:
            val_dataset = None
        
        # Create optimizer
        num_train_steps = len(train_dataset) * epochs
        optimizer, lr_schedule = create_optimizer(
            init_lr=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_train_steps // 10
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss={
                'start_logits': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                'end_logits': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            },
            metrics=['accuracy']
        )
        
        # Train model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=1
        )
        
        return history
    
    def predict(self, question, context):
        """
        Get answer span prediction for a question-context pair.
        
        Args:
            question: Input question
            context: Input context
            
        Returns:
            Predicted answer span
        """
        # Prepare inputs
        inputs = self.prepare_inputs([question], [context])
        
        # Get model predictions
        outputs = self.model(inputs)
        
        # Get predicted answer span
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        start_idx = tf.argmax(start_logits, axis=1)[0]
        end_idx = tf.argmax(end_logits, axis=1)[0]
        
        # Decode answer
        answer = self.tokenizer.decode(
            inputs['input_ids'][0][start_idx:end_idx + 1],
            skip_special_tokens=True
        )
        
        return answer
    
    def evaluate(self, questions, contexts, answers):
        """
        Evaluate model performance using multiple NLP metrics.
        
        Args:
            questions: List of test questions
            contexts: List of test contexts
            answers: List of reference answers
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = []
        exact_matches = []
        f1_scores = []
        bleu_scores = []
        rouge_scores = []
        bert_scores = []
        
        print("\nEvaluating model performance...")
        
        for i, (question, context, answer) in enumerate(zip(questions, contexts, answers)):
            # Get prediction
            pred = self.predict(question, context)
            predictions.append(pred)
            
            # Calculate exact match
            exact_match = 1.0 if pred.strip() == answer.strip() else 0.0
            exact_matches.append(exact_match)
            
            # Calculate F1 score
            pred_tokens = set(word_tokenize(self.preprocess_text(pred)))
            answer_tokens = set(word_tokenize(self.preprocess_text(answer)))
            
            if not pred_tokens or not answer_tokens:
                f1_scores.append(0.0)
            else:
                common = pred_tokens.intersection(answer_tokens)
                if not common:
                    f1_scores.append(0.0)
                else:
                    precision = len(common) / len(pred_tokens)
                    recall = len(common) / len(answer_tokens)
                    f1 = 2 * (precision * recall) / (precision + recall)
                    f1_scores.append(f1)
            
            # Calculate BLEU score
            try:
                reference = [word_tokenize(answer.lower())]
                candidate = word_tokenize(pred.lower())
                bleu = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu)
            except Exception as e:
                print(f"Warning: Could not calculate BLEU score: {e}")
                bleu_scores.append(0.0)
            
            # Calculate ROUGE scores
            if rouge is not None:
                try:
                    rouge_result = rouge.compute(
                        predictions=[pred],
                        references=[answer],
                        use_stemmer=True
                    )
                    rouge_scores.append(rouge_result)
                except Exception as e:
                    print(f"Warning: Could not calculate ROUGE scores: {e}")
            
            # Calculate BERTScore
            if bertscore is not None:
                try:
                    bert_result = bertscore.compute(
                        predictions=[pred],
                        references=[answer],
                        model_type="microsoft/deberta-xlarge-mnli"
                    )
                    bert_scores.append(bert_result['f1'][0])
                except Exception as e:
                    print(f"Warning: Could not calculate BERTScore: {e}")
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(questions)} examples")
        
        # Calculate average metrics
        metrics = {
            'exact_match': np.mean(exact_matches),
            'f1': np.mean(f1_scores),
            'bleu': np.mean(bleu_scores),
            'predictions': predictions
        }
        
        # Add ROUGE scores if available
        if rouge_scores:
            rouge_avg = {
                'rouge1': np.mean([s['rouge1'] for s in rouge_scores]),
                'rouge2': np.mean([s['rouge2'] for s in rouge_scores]),
                'rougeL': np.mean([s['rougeL'] for s in rouge_scores])
            }
            metrics.update(rouge_avg)
        
        # Add BERTScore if available
        if bert_scores:
            metrics['bertscore'] = np.mean(bert_scores)
        
        # Print detailed results
        print("\n=== Evaluation Results ===")
        print(f"Number of examples evaluated: {len(predictions)}")
        print(f"\nExact Match Score: {metrics['exact_match']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"BLEU Score: {metrics['bleu']:.4f}")
        
        if 'rouge1' in metrics:
            print("\nROUGE Scores:")
            print(f"ROUGE-1: {metrics['rouge1']:.4f}")
            print(f"ROUGE-2: {metrics['rouge2']:.4f}")
            print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        
        if 'bertscore' in metrics:
            print(f"\nBERTScore: {metrics['bertscore']:.4f}")
        
        # Print example predictions
        print("\n=== Example Predictions ===")
        for i in range(min(5, len(questions))):
            print(f"\nQuestion: {questions[i]}")
            print(f"Reference Answer: {answers[i]}")
            print(f"Model Prediction: {predictions[i]}")
            print(f"Exact Match: {'✓' if exact_matches[i] == 1.0 else '✗'}")
            print(f"F1 Score: {f1_scores[i]:.4f}")
            print(f"BLEU Score: {bleu_scores[i]:.4f}")
            if rouge_scores:
                print(f"ROUGE-L Score: {rouge_scores[i]['rougeL']:.4f}")
            if bert_scores:
                print(f"BERTScore: {bert_scores[i]:.4f}")
            print("-" * 80)
        
        return metrics
    
    def save(self, path):
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def load(cls, path):
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded ExtractiveQA model
        """
        model = cls()
        model.model = TFAutoModelForQuestionAnswering.from_pretrained(path)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        return model

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    
    Args:
        file_path: Path to the CSV file containing the dataset
        
    Returns:
        Tuple of (train_questions, train_contexts, train_answers,
                 val_questions, val_contexts, val_answers)
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Clean text
    df['Cleaned_Question'] = df['Question'].apply(lambda x: clean_text(x))
    df['Cleaned_Answer'] = df['Answer'].apply(lambda x: clean_text(x))
    
    # Remove empty entries
    df = df.dropna(subset=['Cleaned_Question', 'Cleaned_Answer'])
    df = df[df['Cleaned_Question'].str.len() > 0]
    df = df[df['Cleaned_Answer'].str.len() > 0]
    
    # Split into train and validation sets
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    return (
        train_df['Cleaned_Question'].tolist(),
        train_df['Cleaned_Answer'].tolist(),  # Using answers as context
        train_df['Cleaned_Answer'].tolist(),
        val_df['Cleaned_Question'].tolist(),
        val_df['Cleaned_Answer'].tolist(),  # Using answers as context
        val_df['Cleaned_Answer'].tolist()
    )

def clean_text(text):
    """
    Clean text by removing special characters and extra whitespace.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep question marks
    text = re.sub(r'[^\w\s\?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Example usage
if __name__ == "__main__":
    # Initialize model
    qa_model = ExtractiveQA()
    
    # Load and preprocess data
    train_questions, train_contexts, train_answers, \
    val_questions, val_contexts, val_answers = load_and_preprocess_data('your_data.csv')
    
    # Train model
    history = qa_model.train(
        train_questions, train_contexts, train_answers,
        val_questions, val_contexts, val_answers,
        batch_size=8,
        epochs=3
    )
    
    # Evaluate model
    results = qa_model.evaluate(val_questions, val_contexts, val_answers)
    print("\nEvaluation Results:")
    print(f"Exact Match Score: {results['exact_match']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
    # Save model
    qa_model.save('best_model')