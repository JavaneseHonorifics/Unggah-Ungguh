import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import string

import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rule_based_classifier.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RuleBasedConfig:
    """Configuration class for rule-based classifier settings."""
    
    # Input files
    DICTIONARY_FILE = 'kamus-2cba6-export.json'
    TEST_FILE = 'df_test_group.csv'
    
    # Dictionary structure keys
    DICTIONARY_KEY = 'employees'  # Key in JSON containing dictionary data
    NGOKO_KEY = 'ngoko'
    KRAMA_ALUS_KEY = 'kramaalus'
    KRAMA_INGGIL_KEY = 'kramainggil'
    
    # Classification labels
    LABEL_MAP = {
        "ngoko lugu": 0,
        "ngoko alus": 1,
        "krama lugu": 2,
        "krama alus": 3
    }
    
    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    
    # Special values
    UNKNOWN_LABEL = -1
    
    # Output paths
    RESULTS_SAVE_PATH = 'results/rule_based_results.txt'
    DETAILED_RESULTS_PATH = 'results/rule_based_detailed_results.csv'
    
    # Display settings
    MAX_UNCLASSIFIED_EXAMPLES = 10


class JavaneseDictionaryClassifier:
    """
    Dictionary-based classifier for Javanese text classification.
    
    This classifier uses a dictionary of Javanese words with their speech level
    classifications to determine the overall speech level of input sentences.
    """
    
    def __init__(self, dictionary_path: str, config: RuleBasedConfig = None):
        """
        Initialize the classifier with a dictionary file.
        
        Args:
            dictionary_path: Path to the JSON dictionary file
            config: Configuration object
        """
        self.config = config or RuleBasedConfig()
        self.dictionary = self._load_dictionary(dictionary_path)
        self.classification_stats = {
            'total_sentences': 0,
            'classified_sentences': 0,
            'unclassified_sentences': 0,
            'word_counts': {'ngoko': 0, 'kramaalus': 0, 'kramainggil': 0}
        }
        
        logger.info(f"Initialized classifier with {len(self.dictionary)} dictionary entries")
    
    def _load_dictionary(self, dictionary_path: str) -> Dict:
        """
        Load dictionary from JSON file.
        
        Args:
            dictionary_path: Path to dictionary file
            
        Returns:
            Dictionary data
            
        Raises:
            FileNotFoundError: If dictionary file doesn't exist
            json.JSONDecodeError: If JSON is malformed
            KeyError: If expected dictionary key is missing
        """
        try:
            logger.info(f"Loading dictionary from {dictionary_path}")
            with open(dictionary_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            if self.config.DICTIONARY_KEY not in data:
                raise KeyError(f"Expected key '{self.config.DICTIONARY_KEY}' not found in dictionary")
                
            dictionary_data = data[self.config.DICTIONARY_KEY]
            logger.info(f"Successfully loaded dictionary with {len(dictionary_data)} entries")
            
            return dictionary_data
            
        except FileNotFoundError:
            logger.error(f"Dictionary file not found: {dictionary_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dictionary file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading dictionary: {e}")
            raise
    
    def _preprocess_word(self, word: str) -> str:
        """
        Preprocess word by removing punctuation and converting to lowercase.
        
        Args:
            word: Input word
            
        Returns:
            Preprocessed word
        """
        return word.strip(string.punctuation).lower()
    
    def _count_words_by_type(self, sentence: str) -> Tuple[int, int, int]:
        """
        Count words in sentence by their Javanese speech level type.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Tuple of (ngoko_count, kramaalus_count, kramainggil_count)
        """
        words = sentence.split()
        ngoko_count = 0
        kramaalus_count = 0
        kramainggil_count = 0
        
        for word in words:
            preprocessed_word = self._preprocess_word(word)
            
            # Skip empty words
            if not preprocessed_word:
                continue
                
            # Search in dictionary
            word_found = False
            for entry in self.dictionary.values():
                ngoko_word = entry.get(self.config.NGOKO_KEY, '').lower()
                kramaalus_word = entry.get(self.config.KRAMA_ALUS_KEY, '').lower()
                kramainggil_word = entry.get(self.config.KRAMA_INGGIL_KEY, '').lower()
                
                if preprocessed_word == ngoko_word and ngoko_word:
                    ngoko_count += 1
                    self.classification_stats['word_counts']['ngoko'] += 1
                    word_found = True
                    break
                elif preprocessed_word == kramaalus_word and kramaalus_word:
                    kramaalus_count += 1
                    self.classification_stats['word_counts']['kramaalus'] += 1
                    word_found = True
                    break
                elif preprocessed_word == kramainggil_word and kramainggil_word:
                    kramainggil_count += 1
                    self.classification_stats['word_counts']['kramainggil'] += 1
                    word_found = True
                    break
        
        return ngoko_count, kramaalus_count, kramainggil_count
    
    def _apply_classification_rules(self, ngoko_count: int, 
                                  kramaalus_count: int, 
                                  kramainggil_count: int) -> str:
        """
        Apply classification rules based on word counts.
        
        Args:
            ngoko_count: Number of ngoko words
            kramaalus_count: Number of krama alus words
            kramainggil_count: Number of krama inggil words
            
        Returns:
            Classification string or None if cannot classify
        """
        total_classified_words = ngoko_count + kramaalus_count + kramainggil_count
        
        if total_classified_words == 0:
            return None  # Cannot classify
        
        # Calculate proportions
        ngoko_proportion = ngoko_count / total_classified_words
        kramaalus_proportion = kramaalus_count / total_classified_words
        kramainggil_proportion = kramainggil_count / total_classified_words
        
        # Initial classification based on proportions
        if ngoko_proportion > kramaalus_proportion and ngoko_proportion > kramainggil_proportion:
            classification = "ngoko lugu"
        elif kramaalus_proportion >= ngoko_proportion and kramaalus_proportion > kramainggil_proportion:
            classification = "krama lugu"
        else:
            classification = "krama alus"
        
        # Apply tie-breaking rule: favor ngoko lugu in ties
        if ngoko_proportion == kramaalus_proportion:
            classification = "ngoko lugu"
        
        # Upgrade classification if kramainggil words are present
        if kramainggil_count > 0:
            if classification == "ngoko lugu":
                classification = "ngoko alus"
            elif classification == "krama lugu":
                classification = "krama alus"
        
        # Downgrade classification if ngoko words are present
        if ngoko_count > 0:
            if classification == "krama alus":
                classification = "krama lugu"
            elif classification == "krama lugu":
                classification = "ngoko alus"
        
        return classification
    
    def classify_sentence(self, sentence: str) -> int:
        """
        Classify a single sentence.
        
        Args:
            sentence: Input sentence to classify
            
        Returns:
            Classification label (0-3) or -1 if cannot classify
        """
        if not sentence or not sentence.strip():
            return self.config.UNKNOWN_LABEL
        
        # Count words by type
        ngoko_count, kramaalus_count, kramainggil_count = self._count_words_by_type(sentence)
        
        # Apply classification rules
        classification_str = self._apply_classification_rules(
            ngoko_count, kramaalus_count, kramainggil_count
        )
        
        if classification_str is None:
            return self.config.UNKNOWN_LABEL
        
        return self.config.LABEL_MAP.get(classification_str, self.config.UNKNOWN_LABEL)
    
    def classify_batch(self, sentences: List[str]) -> List[int]:
        """
        Classify a batch of sentences.
        
        Args:
            sentences: List of sentences to classify
            
        Returns:
            List of classification labels
        """
        predictions = []
        
        logger.info(f"Classifying {len(sentences)} sentences...")
        
        for i, sentence in enumerate(sentences):
            if i % 1000 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(sentences)} sentences")
            
            prediction = self.classify_sentence(sentence)
            predictions.append(prediction)
            
            # Update statistics
            self.classification_stats['total_sentences'] += 1
            if prediction != self.config.UNKNOWN_LABEL:
                self.classification_stats['classified_sentences'] += 1
            else:
                self.classification_stats['unclassified_sentences'] += 1
        
        logger.info(f"Classification complete. Classified: {self.classification_stats['classified_sentences']}, "
                   f"Unclassified: {self.classification_stats['unclassified_sentences']}")
        
        return predictions
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        stats = self.classification_stats.copy()
        if stats['total_sentences'] > 0:
            stats['classification_rate'] = stats['classified_sentences'] / stats['total_sentences']
        else:
            stats['classification_rate'] = 0.0
        return stats


class RuleBasedEvaluator:
    """Evaluator for rule-based classifier performance."""
    
    def __init__(self, config: RuleBasedConfig = None):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config or RuleBasedConfig()
    
    def load_test_data(self, test_file_path: str) -> pd.DataFrame:
        """
        Load test data from CSV file.
        
        Args:
            test_file_path: Path to test CSV file
            
        Returns:
            Test DataFrame
        """
        try:
            logger.info(f"Loading test data from {test_file_path}")
            df = pd.read_csv(test_file_path)
            
            # Validate required columns
            required_columns = ['sentence', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Loaded {len(df)} test samples")
            return df
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
    
    def evaluate_classifier(self, classifier: JavaneseDictionaryClassifier, 
                          test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate classifier performance on test data.
        
        Args:
            classifier: Trained classifier
            test_df: Test DataFrame
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting classifier evaluation...")
        
        # Make predictions
        predictions = classifier.classify_batch(test_df['sentence'].tolist())
        test_df = test_df.copy()
        test_df['predicted'] = predictions
        
        # Separate classified and unclassified examples
        unclassified_df = test_df[test_df['predicted'] == self.config.UNKNOWN_LABEL]
        classified_df = test_df[test_df['predicted'] != self.config.UNKNOWN_LABEL]
        
        # Get unclassified examples
        unclassified_count = len(unclassified_df)
        unclassified_examples = unclassified_df['sentence'].tolist()
        
        if len(classified_df) == 0:
            logger.warning("No sentences were classified!")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'confusion_matrix': None,
                'classification_report': None,
                'unclassified_count': unclassified_count,
                'unclassified_examples': unclassified_examples,
                'total_samples': len(test_df),
                'classified_samples': 0
            }
        
        # Calculate metrics on classified samples
        y_true = classified_df['label'].values
        y_pred = classified_df['predicted'].values
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Get unique labels for confusion matrix
        labels = sorted(list(set(y_true) | set(y_pred)))
        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
        
        class_report = classification_report(
            y_true, y_pred, 
            target_names=[self.config.REVERSE_LABEL_MAP.get(i, f'Class_{i}') for i in labels],
            zero_division=0,
            output_dict=True
        )
        
        # Per-class accuracy
        class_accuracies = {}
        for label in labels:
            mask = (y_true == label)
            if mask.sum() > 0:
                class_acc = accuracy_score(y_true[mask], y_pred[mask])
                class_name = self.config.REVERSE_LABEL_MAP.get(label, f'Class_{label}')
                class_accuracies[class_name] = class_acc
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'class_accuracies': class_accuracies,
            'unclassified_count': unclassified_count,
            'unclassified_examples': unclassified_examples,
            'total_samples': len(test_df),
            'classified_samples': len(classified_df),
            'classification_rate': len(classified_df) / len(test_df),
            'labels': labels
        }
        
        logger.info(f"Evaluation complete. Accuracy: {accuracy:.3f}, "
                   f"Classification rate: {results['classification_rate']:.3f}")
        
        return results
    
    def print_results(self, results: Dict[str, Any], classifier_stats: Dict[str, Any]):
        """
        Print evaluation results to console.
        
        Args:
            results: Evaluation results
            classifier_stats: Classifier statistics
        """
        print("\n" + "="*60)
        print("DICTIONARY-BASED RULE CLASSIFIER EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Classified samples: {results['classified_samples']}")
        print(f"  Unclassified samples: {results['unclassified_count']}")
        print(f"  Classification rate: {results['classification_rate']:.3f}")
        
        print(f"\nWord Count Statistics:")
        for word_type, count in classifier_stats['word_counts'].items():
            print(f"  {word_type.capitalize()} words found: {count}")
        
        if results['classified_samples'] > 0:
            print(f"\nPerformance Metrics (on classified samples):")
            print(f"  Accuracy: {results['accuracy']:.3f}")
            print(f"  Precision (weighted): {results['precision']:.3f}")
            print(f"  Recall (weighted): {results['recall']:.3f}")
            print(f"  F1-score (weighted): {results['f1_score']:.3f}")
            
            print(f"\nPer-class Accuracy:")
            for class_name, acc in results['class_accuracies'].items():
                print(f"  {class_name}: {acc:.3f}")
            
            print(f"\nConfusion Matrix:")
            print(results['confusion_matrix'])
            
            print(f"\nDetailed Classification Report:")
            # Print classification report in readable format
            report = results['classification_report']
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    print(f"  {class_name}:")
                    print(f"    Precision: {metrics['precision']:.3f}")
                    print(f"    Recall: {metrics['recall']:.3f}")
                    print(f"    F1-score: {metrics['f1-score']:.3f}")
                    print(f"    Support: {metrics['support']}")
        
        if results['unclassified_count'] > 0:
            print(f"\nUnclassified Examples (showing up to {self.config.MAX_UNCLASSIFIED_EXAMPLES}):")
            for i, example in enumerate(results['unclassified_examples'][:self.config.MAX_UNCLASSIFIED_EXAMPLES]):
                print(f"  {i+1}. {example}")
            
            if len(results['unclassified_examples']) > self.config.MAX_UNCLASSIFIED_EXAMPLES:
                remaining = len(results['unclassified_examples']) - self.config.MAX_UNCLASSIFIED_EXAMPLES
                print(f"  ... and {remaining} more unclassified examples")
    
    def save_results(self, results: Dict[str, Any], classifier_stats: Dict[str, Any]):
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results
            classifier_stats: Classifier statistics
        """
        # Create results directory
        os.makedirs(os.path.dirname(self.config.RESULTS_SAVE_PATH), exist_ok=True)
        
        # Save summary results
        with open(self.config.RESULTS_SAVE_PATH, 'w') as f:
            f.write("Dictionary-Based Rule Classifier Results\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Dataset Statistics:\n")
            f.write(f"Total samples: {results['total_samples']}\n")
            f.write(f"Classified samples: {results['classified_samples']}\n")
            f.write(f"Unclassified samples: {results['unclassified_count']}\n")
            f.write(f"Classification rate: {results['classification_rate']:.3f}\n\n")
            
            f.write(f"Word Count Statistics:\n")
            for word_type, count in classifier_stats['word_counts'].items():
                f.write(f"{word_type.capitalize()} words found: {count}\n")
            f.write("\n")
            
            if results['classified_samples'] > 0:
                f.write(f"Performance Metrics:\n")
                f.write(f"Accuracy: {results['accuracy']:.3f}\n")
                f.write(f"Precision (weighted): {results['precision']:.3f}\n")
                f.write(f"Recall (weighted): {results['recall']:.3f}\n")
                f.write(f"F1-score (weighted): {results['f1_score']:.3f}\n\n")
                
                f.write(f"Per-class Accuracy:\n")
                for class_name, acc in results['class_accuracies'].items():
                    f.write(f"{class_name}: {acc:.3f}\n")
                f.write("\n")
                
                f.write(f"Confusion Matrix:\n")
                f.write(str(results['confusion_matrix']) + "\n\n")
            
            f.write(f"Unclassified Examples:\n")
            for i, example in enumerate(results['unclassified_examples']):
                f.write(f"{i+1}. {example}\n")
        
        logger.info(f"Results saved to {self.config.RESULTS_SAVE_PATH}")


def rule_based_model(args):
    """Main execution function."""
    config = RuleBasedConfig()
    
    # Override config with command line arguments
    if args.dictionary_file:
        config.DICTIONARY_FILE = args.dictionary_file
    if args.test_file:
        config.TEST_FILE = args.test_file
    
    logger.info("Starting Dictionary-Based Rule Classifier evaluation")
    
    try:
        # Initialize classifier
        classifier = JavaneseDictionaryClassifier(config.DICTIONARY_FILE, config)
        
        # Initialize evaluator
        evaluator = RuleBasedEvaluator(config)
        
        # Load test data
        test_df = evaluator.load_test_data(config.TEST_FILE)
        
        # Evaluate classifier
        results = evaluator.evaluate_classifier(classifier, test_df)
        
        # Get classifier statistics
        classifier_stats = classifier.get_classification_stats()
        
        # Print and save results
        evaluator.print_results(results, classifier_stats)
        
        if args.save_results:
            evaluator.save_results(results, classifier_stats)
        
        logger.info("Evaluation completed successfully")
        
        return classifier, results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise