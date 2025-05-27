import argparse
import logging
import sys
from src.model.Rule_based import rule_based_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rule_based_classifier.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Dictionary-Based Rule Classifier for Javanese Text')
    
    parser.add_argument('--dictionary-file', type=str,
                       help='Path to the dictionary JSON file')
    parser.add_argument('--test-file', type=str,
                       help='Path to the test CSV file')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    classifier, results = rule_based_model(args)