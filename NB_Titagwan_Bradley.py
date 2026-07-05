import os
import re
import math
import time
import itertools

import numpy as np
import matplotlib.pyplot as plt
import nltk


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize stopword list and lemmatizer
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Different preprocessing configurations to evaluate model performance
# Each config toggles options like punctuation removal, stopword removal, and smoothing parameter alpha
# This preprocessing options are used to increase the quality of the training data.
PREPROCESS_OPTIONS = [
    {
        'name': 'lower_only',
        'lower': True,
        'remove_punct': False,
        'remove_stop': False,
        'min_len': 2,
        'alpha': 1.0,
    },
    {
        'name': 'lower_punct',
        'lower': True,
        'remove_punct': True,
        'remove_stop': False,
        'min_len': 2,
        'alpha': 1.0,
    },
    {
        'name': 'lower_punct_stop',
        'lower': True,
        'remove_punct': True,
        'remove_stop': True,
        'min_len': 2,
        'alpha': 1.0,
    },
    {
        'name': 'lower_punct_stop_alpha05',
        'lower': True,
        'remove_punct': True,
        'remove_stop': True,
        'min_len': 2,
        'alpha': 0.5,
    },
    {
        'name': 'lower_punct_stop_alpha15',
        'lower': True,
        'remove_punct': True,
        'remove_stop': True,
        'min_len': 2,
        'alpha': 1.5,
    },
]


def preprocess_text(text, lower=True, remove_punct=True, remove_stop=True, min_len=2, lemmatize=True):
    """
    Preprocess a sentence by:
    - Lowercasing text
    - Removing punctuation
    - Tokenizing into words
    - Removing stopwords
    - Lemmatizing words (reducing to base form)
    - Filtering out short tokens
    
    Returns:
        List of cleaned tokens
    """
    if lower:
        text = text.lower()

    if remove_punct:
        text = re.sub(r'[^a-z\s]', ' ', text)

    tokens = text.split()

    if remove_stop:
        tokens = [token for token in tokens if token not in STOP_WORDS]

    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    tokens = [token for token in tokens if len(token) >= min_len]
    return tokens


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier for text classification.
    
    Uses:
    - Log probabilities to avoid underflow
    - Laplace smoothing (controlled by alpha)
    """
    def __init__(self, alpha=1.0):
        # Smoothing parameter
        self.alpha = alpha # Laplace smoothing parameter

        # Data structures to store model statistics
        self.labels = []                    # List of class labels
        self.vocab = set()                  # Vocabulary (unique words)
        self.class_doc_counts = {}          # Number of documents per class
        self.class_total_tokens = {}        # Total word count per class
        self.class_word_counts = {}         # Word frequency per class
        self.log_priors = {}                # Log prior probabilities
        self.vocab_size = 0                 # Size of vocabulary

    def train(self, documents, labels):
        """
        Train the Naive Bayes model by:
        - Counting documents per class
        - Counting word frequencies per class
        - Building vocabulary
        - Computing prior probabilities
        """  
        self.labels = sorted(list(set(labels)))# Get unique class labels
        self.vocab = set()                      # Reset vocabulary
        self.class_doc_counts = {label: 0 for label in self.labels}
        self.class_total_tokens = {label: 0 for label in self.labels}
        self.class_word_counts = {label: {} for label in self.labels}
        self.log_priors = {}

        # The label are the categories and the tokens are the text extracted from the sentences
        # Loop through each document
        for tokens, label in zip(documents, labels):
            # Count documents per class
            self.class_doc_counts[label] += 1

            # Loop through words
            for token in tokens:
                # Add to vocabulary
                self.vocab.add(token)
                # Count total tokens
                self.class_total_tokens[label] += 1
                if token not in self.class_word_counts[label]:
                    # Initialize count
                    self.class_word_counts[label][token] = 0
                # Increment word count    
                self.class_word_counts[label][token] += 1
        
        # Total number of documents
        total_docs = len(labels)
        # Vocabulary size
        self.vocab_size = len(self.vocab)

        for label in self.labels:
            # Compute log prior probability for each class
            self.log_priors[label] = math.log(self.class_doc_counts[label] / total_docs)

    def predict_log_proba_one(self, tokens):
        """
        Compute log-probabilities for each class given a document.
        
        Uses:
        log P(class) + sum(log P(word | class))
        """
        # Dictionary to store scores for each class
        scores = {}
        for label in self.labels:
            # Start with prior probability
            score = self.log_priors[label]
            # Denominator for Laplace smoothing
            denominator = self.class_total_tokens[label] + self.alpha * self.vocab_size

            for token in tokens:
                # Word frequency
                count = self.class_word_counts[label].get(token, 0)
                # Smoothed probability
                probability = (count + self.alpha) / denominator
                # Add log probability
                score += math.log(probability)
            
            # Store score for class
            scores[label] = score
        return scores

    def predict_one(self, tokens):
        """
        Predict the most likely class for a single document.
        Returns:
            best label and all class scores
        """
        # Get scores for all classes      
        scores = self.predict_log_proba_one(tokens)
        # Choose class with highest score
        best_label = max(scores, key=scores.get)
        # Return prediction and scores
        return best_label, scores

    def predict(self, documents):
        """
        Predict labels for multiple documents.
        Returns:
            list of predictions and corresponding score dictionaries
        """
        predictions = []
        score_list = []

        # Predict one document, store prediction and store scores
        for tokens in documents:
            pred, scores = self.predict_one(tokens)
            predictions.append(pred)
            score_list.append(scores)
        return predictions, score_list


# ----------------- DATA LOADERS -----------------

def load_training_txt_files(directory_path):
    """
    Load training data from multiple text files.
    Each file corresponds to a specific class.
    
    Returns:
        texts (list of sentences)
        labels (list of class labels)
    """

    files = {
        'training_physics.txt': 'physics',
        'training_biology.txt': 'biology',
        'training_history.txt': 'world history',
        'training_economics.txt': 'economics',
    }
    texts = []
    labels = []

    for filename, label in files.items():
        file_path = os.path.join(directory_path, filename)
        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found in {directory_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip()
                if sentence:
                    texts.append(sentence)
                    labels.append(label)

    return texts, labels



def load_testing_txt_file(directory_path):
    """
    Load test data where each line is formatted as:
    label: sentence
    
    Returns:
        texts and labels
    """
     
    texts = []
    labels = []
    file_path = os.path.join(directory_path, 'testing_data.txt')

    if not os.path.exists(file_path):
        print(f"Warning: testing_data.txt not found in {directory_path}")
        return texts, labels

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line and ':' in clean_line:
                label, sentence = clean_line.split(':', 1)
                texts.append(sentence.strip())
                labels.append(label.strip().lower())

    return texts, labels



def parse_three_column_csv_line(line):
    """
    Safely parse a CSV line with format:
    Id,Comment,Topic
    
    Handles commas inside quoted text.
    Returns:
        (comment, topic) or None if invalid
    """

    line = line.rstrip('\n').rstrip('\r')
    if not line:
        return None

    if ',' not in line:
        return None

    left_split = line.split(',', 1)
    if len(left_split) != 2:
        return None

    _, remainder = left_split
    if ',' not in remainder:
        return None

    comment, topic = remainder.rsplit(',', 1)
    comment = comment.strip()
    topic = topic.strip().strip('"').lower()

    if comment.startswith('"') and comment.endswith('"') and len(comment) >= 2:
        comment = comment[1:-1]
    comment = comment.replace('""', '"').strip()

    if not comment or not topic:
        return None

    return comment, topic



def load_kaggle_csv(file_path):
    """
    Load dataset from CSV file.
    Extracts comment text and topic labels.
    """
    texts = []
    labels = []

    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return texts, labels

    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if not first_line:
            return texts, labels

        for line in f:
            parsed = parse_three_column_csv_line(line)
            if parsed is None:
                continue
            comment, topic = parsed
            texts.append(comment)
            labels.append(topic)

    return texts, labels



def split_first_1200_rest(texts, labels):
    """
    Split dataset into:
    - First 1200 samples for training
    - Remaining samples for testing
    """
    split_index = min(1200, len(texts))
    return texts[:split_index], labels[:split_index], texts[split_index:], labels[split_index:]


# ----------------- EVALUATION -----------------

def build_confusion_matrix(true_labels, predicted_labels, classes):
    """
    Build confusion matrix:
    Rows = true labels
    Columns = predicted labels
    """

    # Map labels to indices
    index = {label: i for i, label in enumerate(classes)}
    # Map labels to indices
    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for true_label, pred_label in zip(true_labels, predicted_labels):
        if true_label in index and pred_label in index:
            # Increment cell
            matrix[index[true_label], index[pred_label]] += 1
    return matrix



def accuracy_score(true_labels, predicted_labels):
    """
    Compute classification accuracy.
    """
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    return correct / len(true_labels) if true_labels else 0.0



def precision_recall_f1(confusion, classes):
    """
    Compute precision, recall, and F1-score for each class.
    """
    metrics = {}
    for i, label in enumerate(classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    return metrics



def most_confused_pairs(confusion, classes, top_n=6):
    """
    Identify class pairs that are most frequently confused.
    """
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and confusion[i, j] > 0:
                pairs.append((confusion[i, j], classes[i], classes[j]))
    pairs.sort(reverse=True)
    return pairs[:top_n]



def explain_misclassification(text, true_label, pred_label):
    """
    Provide heuristic explanation for why a sentence was misclassified.
    """
    lowered = text.lower()
    reasons = []

    if len(lowered.split()) <= 6:
        reasons.append('sentence is short, so there are fewer clues for the classifier')

    shared_word_map = {
        'biology': ['energy', 'system', 'cell', 'organism', 'genetic'],
        'physics': ['energy', 'force', 'motion', 'field', 'system'],
        'economics': ['market', 'trade', 'policy', 'growth', 'value'],
        'world history': ['war', 'empire', 'global', 'society', 'revolution'],
    }

    shared_hits = []
    for word in shared_word_map.get(true_label, []):
        if word in lowered:
            shared_hits.append(word)
    for word in shared_word_map.get(pred_label, []):
        if word in lowered and word not in shared_hits:
            shared_hits.append(word)

    if shared_hits:
        reasons.append('contains broad words that appear in more than one class: ' + ', '.join(shared_hits[:4]))

    if not reasons:
        reasons.append('important keywords were rare in training or overlapped across classes')

    return '; '.join(reasons)



def find_misclassifications(test_texts, true_labels, predicted_labels, score_list, max_examples=8):
    """
    Collect misclassified examples along with prediction scores and explanations.
    """
    mistakes = []
    for text, true_label, pred_label, scores in zip(test_texts, true_labels, predicted_labels, score_list):
        if true_label != pred_label:
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            mistakes.append({
                'text': text,
                'true': true_label,
                'pred': pred_label,
                'scores': sorted_scores,
                'reason': explain_misclassification(text, true_label, pred_label),
            })
    return mistakes[:max_examples]



def plot_confusion_matrix(confusion, classes, title, filename):
    """
    Plot and save confusion matrix as an image.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    threshold = confusion.max() / 2.0 if confusion.size else 0.0
    for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(
            j,
            i,
            str(int(confusion[i, j])),
            ha='center',
            va='center',
            color='white' if confusion[i, j] > threshold else 'black',
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()



def plot_accuracy_summary(results, filename, title):
    """
    Plot accuracy comparison across preprocessing configurations.
    """
    names = [result['config']['name'] for result in results]
    accuracies = [result['accuracy'] for result in results]

    plt.figure(figsize=(9, 5))
    plt.plot(names, accuracies, marker='o')
    plt.ylabel('Accuracy')
    plt.xlabel('Preprocessing setting')
    plt.title(title)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()



def print_confusion_matrix(confusion, classes):
    """
    Plot and save confusion matrix as an image.
    """
    header = 'true\\pred'.ljust(18)
    for label in classes:
        header += label[:16].ljust(18)
    print(header)

    for i, row_label in enumerate(classes):
        row = row_label[:16].ljust(18)
        for j in range(len(classes)):
            row += str(confusion[i, j]).ljust(18)
        print(row)



def print_metrics(metrics):
    print('\nPer-class metrics:')
    for label, values in metrics.items():
        print(
            f"  {label:<16} precision={values['precision']:.4f}  "
            f"recall={values['recall']:.4f}  f1={values['f1']:.4f}"
        )



def print_confused_pairs(pairs):
    print('\nMost confused class pairs:')
    if not pairs:
        print('  None')
        return
    for count, true_label, pred_label in pairs:
        print(f'  true={true_label:<16} predicted={pred_label:<16} count={count}')



def print_misclassifications(mistakes, dataset_label="DATASET"):
    print(f'\n{dataset_label}: Example misclassifications:')
    if not mistakes:
        print(f"  {dataset_label}: None")
        return

    for i, item in enumerate(mistakes, start=1):
        score_text = ', '.join([f'{label}={score:.3f}' for label, score in item['scores']])
        print(f"{dataset_label}: {i}. true={item['true']} | predicted={item['pred']}")
        print(f"     sentence: {item['text']}")
        print(f"     log-scores: {score_text}")
        print(f"     reason: {item['reason']}")



def run_experiment(train_texts, train_labels, test_texts, test_labels, config, prefix, allowed_classes=None):

    """
    Run a single experiment:
    - Preprocess data
    - Train model
    - Predict results
    - Compute evaluation metrics
    - Save confusion matrix
    
    Returns:
        Dictionary of results
    """

    # Preprocess training data
    processed_train = [
        preprocess_text(
            text,
            lower=config['lower'],
            remove_punct=config['remove_punct'],
            remove_stop=config['remove_stop'],
            min_len=config['min_len'],
            lemmatize=True,
        )
        for text in train_texts
    ]

    # Preprocess testing data
    processed_test = [
        preprocess_text(
            text,
            lower=config['lower'],
            remove_punct=config['remove_punct'],
            remove_stop=config['remove_stop'],
            min_len=config['min_len'],
            lemmatize=True,
        )
        for text in test_texts
    ]


    # Start timer
    start = time.time()
    # Initialize model
    model = NaiveBayesClassifier(alpha=config['alpha'])
    # Train model
    model.train(processed_train, train_labels)
    # Predict
    predictions, score_list = model.predict(processed_test)
    # Measure runtime
    elapsed = time.time() - start

    # Use only allowed classes if specified
    if allowed_classes is not None:
        classes = allowed_classes
    else:
        classes = sorted(list(set(train_labels + test_labels)))

    confusion = build_confusion_matrix(test_labels, predictions, classes)
    metrics = precision_recall_f1(confusion, classes)
    confused_pairs = most_confused_pairs(confusion, classes)
    mistakes = find_misclassifications(test_texts, test_labels, predictions, score_list)
    accuracy = accuracy_score(test_labels, predictions)

    image_name = f'{prefix}_confusion_{config["name"]}.png'
    plot_confusion_matrix(confusion, classes, f'Confusion Matrix: {prefix} - {config["name"]}', image_name)

    return {
        'config': config,
        'accuracy': accuracy,
        'elapsed': elapsed,
        'vocab_size': model.vocab_size,
        'classes': classes,
        'confusion': confusion,
        'metrics': metrics,
        'confused_pairs': confused_pairs,
        'mistakes': mistakes,
        'image_name': image_name,
    }


def run_dataset(train_texts, train_labels, test_texts, test_labels, prefix, dataset_title, dataset_label, allowed_classes=None):
    """
    Run experiments across all preprocessing configurations.
    
    For each config:
    - Train model
    - Evaluate performance
    - Print results
    
    Also:
    - Finds best configuration
    - Generates accuracy comparison plot
    """
    print('\n' + '=' * 80)
    print(f"{dataset_label}: {dataset_title}")
    print(f"{dataset_label}: Loaded training sentences: {len(train_texts)}")
    print(f"{dataset_label}: Loaded testing sentences: {len(test_texts)}")

    results = []
    for config in PREPROCESS_OPTIONS:
        print('\n' + '-' * 60)
        print(f"Running setting: {config['name']}")
        result = run_experiment(train_texts, train_labels, test_texts, test_labels, config, prefix, allowed_classes=allowed_classes)
        results.append(result)

        print(f"\n{dataset_label}: Running setting: {config['name']}")
        print(f"{dataset_label}: Accuracy: {result['accuracy']:.4f}")
        print(f"{dataset_label}: Runtime: {result['elapsed']:.6f} seconds")
        print(f"{dataset_label}: Vocabulary size: {result['vocab_size']}")
        print(f"{dataset_label}: Confusion matrix image saved to: {result['image_name']}")
        print_confusion_matrix(result['confusion'], result['classes'])
        print_metrics(result['metrics'])
        print_confused_pairs(result['confused_pairs'])
        print_misclassifications(result['mistakes'][:6], dataset_label=dataset_label)

    best_result = max(results, key=lambda item: item['accuracy'])

    print('\n' + '#' * 80)
    print(f"{dataset_label}: Best setting summary")
    print(f"{dataset_label}: Best setting: {best_result['config']['name']}")
    print(f"{dataset_label}: Best accuracy: {best_result['accuracy']:.4f}")
    print(f"{dataset_label}: Best confusion matrix image: {best_result['image_name']}")


    summary_plot = f'{prefix}_accuracy_comparison.png'
    plot_accuracy_summary(results, summary_plot, f'Accuracy Comparison: {dataset_title}')
    print(f"Accuracy comparison plot saved to: {summary_plot}")

    return results



def main():
    """
    Entry point of the program.
    
    Steps:
    1. Load provided dataset
    2. Run experiments on provided dataset
    3. Load Kaggle dataset
    4. Split into train/test
    5. Run experiments on extra dataset
    """
    base_dir = './'
    KAGGLE_ALLOWED_CLASSES = ['biology', 'chemistry', 'physics']
    
    # Load provided dataset
    provided_train_texts, provided_train_labels = load_training_txt_files(base_dir)
    provided_test_texts, provided_test_labels = load_testing_txt_file(base_dir)
    # Run experiments on provided dataset
    run_dataset(
        provided_train_texts,
        provided_train_labels,
        provided_test_texts,
        provided_test_labels,
        prefix='provided',
        dataset_title='RUNNING PROVIDED DATASET',
        dataset_label='PROVIDED'
    )

    extra_file = os.path.join(base_dir, 'train.csv')
    extra_texts, extra_labels = load_kaggle_csv(extra_file)
    extra_train_texts, extra_train_labels, extra_test_texts, extra_test_labels = split_first_1200_rest(extra_texts, extra_labels)

    if extra_train_texts and extra_test_texts:
        run_dataset(
            extra_train_texts,
            extra_train_labels,
            extra_test_texts,
            extra_test_labels,
            prefix='extra',
            dataset_title='RUNNING EXTRA DATASET: FIRST 1200 TRAIN / REST TEST',
            dataset_label='KAGGLE',
            allowed_classes=KAGGLE_ALLOWED_CLASSES  # <-- restrict to biology, chemistry, physics
        )
    else:
        print('\nWarning: extra dataset could not be split into train and test sets.')


if __name__ == '__main__':
    # Run the entire pipeline
    main()
