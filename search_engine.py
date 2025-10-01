#!/usr/bin/env python3
"""
Search Engine with Auto-Correct using BPE Tokenization

This module implements a sophisticated search engine that:
1. Uses Byte Pair Encoding (BPE) for subword tokenization
2. Provides intelligent auto-correct for misspelled queries
3. Offers interactive user selection for multiple correction options
4. Handles out-of-vocabulary words gracefully

Author: AI Assistant
Date: 2024
"""

import os
import re
import math
from collections import Counter, defaultdict

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

# Import BPE tokenization library with graceful fallback
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("Warning: 'tokenizers' library not found. Install with: pip install tokenizers")


# ============================================================================
# PART 1: TEXT PROCESSING AND CORPUS LOADING
# ============================================================================

def split_into_paragraphs(text):
    """
    Split raw text into paragraphs based on blank lines.
    
    Args:
        text (str): Raw text content from a document
        
    Returns:
        list: List of paragraph strings (non-empty only)
    """
    # Split on one or more blank lines, strip whitespace, filter empty strings
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', text) if p.strip()]
    return paragraphs


def load_corpus(corpus_dir, sample_n=None):
    """
    Load text files from a directory and convert them into searchable paragraphs.
    
    Args:
        corpus_dir (str): Path to directory containing .txt files
        sample_n (int, optional): Number of files to sample for testing
        
    Returns:
        list: List of dictionaries, each containing:
            - 'id': Unique paragraph identifier
            - 'book': Source book name
            - 'text': Paragraph content
    """
    paragraphs = []
    
    # Get all .txt files in the directory
    files = [f for f in os.listdir(corpus_dir) if f.endswith('.txt')]
    
    # Sample files if requested (for faster testing)
    if sample_n:
        import random
        random.seed(42)  # For reproducible results
        files = random.sample(files, min(sample_n, len(files)))
    
    # Process each file
    para_count = 0
    for fname in files:
        book_name = os.path.splitext(fname)[0]  # Remove .txt extension
        file_path = os.path.join(corpus_dir, fname)
        
        # Read file with error handling for encoding issues
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file_handle:
            text_content = file_handle.read()
        
        # Split into paragraphs
        file_paragraphs = split_into_paragraphs(text_content)
        
        # Create paragraph records
        for paragraph_text in file_paragraphs:
            para_id = f"{book_name}__para{para_count:06d}"
            paragraphs.append({
                'id': para_id,
                'book': book_name,
                'text': paragraph_text
            })
            para_count += 1
    
    print(f"Loaded {len(paragraphs)} paragraphs from {len(files)} books.")
    return paragraphs


# ============================================================================
# PART 2: BPE TOKENIZATION SYSTEM
# ============================================================================

class BPETokenizer:
    """
    Wrapper class for HuggingFace tokenizers BPE implementation.
    
    This class provides a simplified interface for training and using BPE tokenization
    on our corpus, with automatic subword learning and out-of-vocabulary handling.
    """
    
    def __init__(self, vocab_size=5000):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size (int): Maximum vocabulary size for the tokenizer
        """
        self.vocab_size = vocab_size
        self.tokenizer = None
    
    def fit(self, texts):
        """
        Train the BPE tokenizer on a list of texts.
        
        Args:
            texts (list): List of text strings to train on
        """
        print(f"Training BPE tokenizer with vocab_size={self.vocab_size}...")
        
        # Initialize tokenizer with BPE model
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()  # Split on whitespace first
        
        # Configure trainer with special tokens
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[PAD]"],  # Unknown and padding tokens
            show_progress=True
        )
        
        # Write texts to temporary file for training
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            for text in texts:
                temp_file.write(text.lower() + '\n')  # Convert to lowercase
            temp_file_path = temp_file.name
        
        # Train the tokenizer
        self.tokenizer.train([temp_file_path], trainer)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        print(f"BPE training complete. Vocabulary size: {self.tokenizer.get_vocab_size()}")
    
    def encode(self, text):
        """
        Encode text into BPE tokens.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            list: List of BPE tokens
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call fit() first.")
        return self.tokenizer.encode(text.lower()).tokens
    
    def get_vocab(self):
        """
        Get the vocabulary as a set of tokens.
        
        Returns:
            set: Set of all vocabulary tokens
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call fit() first.")
        return set(self.tokenizer.get_vocab().keys())


# ============================================================================
# PART 3: INVERTED INDEX CONSTRUCTION
# ============================================================================

def build_inverted_index(paragraphs, tokenizer):
    """
    Build an inverted index for fast text search.
    
    An inverted index maps each token to a list of paragraph IDs that contain it.
    This allows for efficient retrieval of documents containing specific terms.
    
    Args:
        paragraphs (list): List of paragraph dictionaries
        tokenizer: Trained BPE tokenizer
        
    Returns:
        tuple: (inverted_index, vocab, paragraph_tokens)
            - inverted_index: dict mapping tokens to paragraph ID lists
            - vocab: set of all vocabulary tokens
            - paragraph_tokens: dict mapping paragraph IDs to their token lists
    """
    inverted_index = defaultdict(list)
    vocab = tokenizer.get_vocab()
    paragraph_tokens = {}  # Store tokens for each paragraph for ranking
    
    print("Building inverted index...")
    
    # Process each paragraph
    for i, paragraph in enumerate(paragraphs):
        # Progress indicator for large corpora
        if (i + 1) % 5000 == 0:
            print(f"  Indexed {i + 1} paragraphs...")
        
        para_id = paragraph['id']
        
        # Tokenize the paragraph text
        tokens = tokenizer.encode(paragraph['text'])
        paragraph_tokens[para_id] = tokens
        
        # Add each unique token to the inverted index
        for token in set(tokens):
            inverted_index[token].append(para_id)
    
    print(f"Inverted index complete. Vocabulary size: {len(vocab)}")
    return inverted_index, vocab, paragraph_tokens


# ============================================================================
# PART 4: AUTO-CORRECT SYSTEM (EDIT DISTANCE)
# ============================================================================

def levenshtein_distance(a, b, max_dist=3):
    """
    Compute Levenshtein distance between two strings with early stopping.
    
    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one string into another.
    
    Args:
        a (str): First string
        b (str): Second string
        max_dist (int): Maximum distance to compute (for efficiency)
        
    Returns:
        int: Edit distance between strings (or max_dist + 1 if exceeded)
    """
    m, n = len(a), len(b)
    
    # Early termination if length difference exceeds max distance
    if abs(m - n) > max_dist:
        return max_dist + 1
    
    # Dynamic programming table (only need previous row)
    prev = list(range(n + 1))
    
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        min_in_row = cur[0]
        
        for j in range(1, n + 1):
            # Cost of substitution (0 if characters match, 1 otherwise)
            cost = 0 if a[i - 1] == b[j - 1] else 1
            
            # Minimum of: deletion, insertion, substitution
            cur[j] = min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
                prev[j - 1] + cost  # substitution
            )
            
            # Track minimum in current row for early stopping
            if cur[j] < min_in_row:
                min_in_row = cur[j]
        
        # Early termination if minimum distance exceeds threshold
        if min_in_row > max_dist:
            return max_dist + 1
        
        prev = cur
    
    return prev[n]


def suggest_corrections(token, vocab, max_dist=2, max_suggestions=5):
    """
    Find closest vocabulary words within edit distance threshold.
    
    Args:
        token (str): Misspelled token to correct
        vocab (set): Set of vocabulary words
        max_dist (int): Maximum edit distance to consider
        max_suggestions (int): Maximum number of suggestions to return
        
    Returns:
        list: List of (word, distance) tuples, sorted by distance
    """
    suggestions = []
    
    # Calculate edit distance to all vocabulary words
    for vocab_word in vocab:
        distance = levenshtein_distance(token, vocab_word, max_dist)
        if distance <= max_dist:
            suggestions.append((vocab_word, distance))
    
    # Sort by distance, then by word length (prefer shorter words for same distance)
    suggestions.sort(key=lambda x: (x[1], len(x[0])))
    
    return suggestions[:max_suggestions]


def suggest_correction(token, vocab, max_dist=2):
    """
    Find the single closest vocabulary word (backward compatibility).
    
    Args:
        token (str): Misspelled token to correct
        vocab (set): Set of vocabulary words
        max_dist (int): Maximum edit distance to consider
        
    Returns:
        tuple: (best_word, distance) or (None, None) if no match found
    """
    suggestions = suggest_corrections(token, vocab, max_dist, 1)
    if suggestions:
        return suggestions[0]
    return None, None


# ============================================================================
# PART 5: VOCABULARY EXTRACTION FOR AUTO-CORRECT
# ============================================================================

def extract_whole_words_from_corpus(paragraphs):
    """
    Extract whole words from the original corpus text for auto-correct.
    
    This creates a vocabulary of complete words (not BPE subwords) that can be
    used for auto-correcting misspelled queries.
    
    Args:
        paragraphs (list): List of paragraph dictionaries
        
    Returns:
        set: Set of unique whole words from the corpus
    """
    whole_words = set()
    
    for paragraph in paragraphs:
        # Extract words using regex (letters only, case-insensitive)
        words = re.findall(r'\b[a-z]+\b', paragraph['text'].lower())
        
        # Add words to vocabulary (minimum 2 characters)
        for word in words:
            if len(word) > 1:
                whole_words.add(word)
    
    return whole_words


def break_into_subwords(word, vocab, tokenizer):
    """
    Break an out-of-vocabulary word into known subwords using BPE.
    
    Args:
        word (str): Word to break down
        vocab (set): BPE vocabulary
        tokenizer: Trained BPE tokenizer
        
    Returns:
        list: List of known subword tokens
    """
    tokens = tokenizer.encode(word)
    # Filter to only known tokens (not UNK)
    known_tokens = [t for t in tokens if t != '[UNK]' and t in vocab]
    return known_tokens


# ============================================================================
# PART 6: SEARCH AND RANKING
# ============================================================================

def cosine_similarity(query_tokens, para_tokens):
    """
    Calculate cosine similarity between query and paragraph token vectors.
    
    Cosine similarity measures the angle between two vectors, providing a
    normalized similarity score between 0 and 1.
    
    Args:
        query_tokens (list): List of query tokens
        para_tokens (list): List of paragraph tokens
        
    Returns:
        float: Cosine similarity score (0.0 to 1.0)
    """
    # Count token frequencies
    query_counter = Counter(query_tokens)
    para_counter = Counter(para_tokens)
    
    # Calculate dot product (sum of products of corresponding frequencies)
    dot_product = sum(query_counter[token] * para_counter[token] for token in query_counter)
    
    # Calculate vector magnitudes
    query_magnitude = math.sqrt(sum(count * count for count in query_counter.values()))
    para_magnitude = math.sqrt(sum(count * count for count in para_counter.values()))
    
    # Avoid division by zero
    if query_magnitude == 0 or para_magnitude == 0:
        return 0.0
    
    # Return normalized cosine similarity
    return dot_product / (query_magnitude * para_magnitude)


def search_query(query_tokens, inverted_index, paragraph_tokens, paragraphs, top_k=10):
    """
    Search for paragraphs matching query tokens using inverted index.
    
    Args:
        query_tokens (list): List of tokens to search for
        inverted_index (dict): Inverted index mapping tokens to paragraph IDs
        paragraph_tokens (dict): Mapping of paragraph IDs to their token lists
        paragraphs (list): List of all paragraph dictionaries
        top_k (int): Maximum number of results to return
        
    Returns:
        list: Ranked list of search results, each containing:
            - 'id': Paragraph identifier
            - 'book': Source book name
            - 'text': Paragraph content
            - 'score': Similarity score
    """
    # Step 1: Find all paragraphs containing any query token
    candidate_paragraphs = set()
    for token in query_tokens:
        if token in inverted_index:
            candidate_paragraphs.update(inverted_index[token])
    
    # Return empty list if no candidates found
    if not candidate_paragraphs:
        return []
    
    # Step 2: Calculate similarity scores for all candidates
    results = []
    para_dict = {p['id']: p for p in paragraphs}  # For fast lookup
    
    for para_id in candidate_paragraphs:
        paragraph = para_dict[para_id]
        tokens = paragraph_tokens[para_id]
        
        # Calculate cosine similarity between query and paragraph
        similarity = cosine_similarity(query_tokens, tokens)
        
        # Store result with metadata
        results.append({
            'id': para_id,
            'book': paragraph['book'],
            'text': paragraph['text'],
            'score': similarity
        })
    
    # Step 3: Sort by similarity score (descending) and return top-k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


# ============================================================================
# PART 7: MAIN QUERY PROCESSING WITH AUTO-CORRECT
# ============================================================================

def process_query(query, vocab, whole_words, tokenizer, inverted_index, paragraph_tokens, paragraphs, threshold=2):
    """
    Process a search query with intelligent auto-correct and OOV handling.
    
    This is the main function that orchestrates the entire search process:
    1. Tokenizes the input query
    2. Checks each word against the vocabulary
    3. Applies auto-correct for misspelled words
    4. Handles out-of-vocabulary words with BPE subwords
    5. Executes the search and returns results
    
    Args:
        query (str): User's search query
        vocab (set): BPE vocabulary
        whole_words (set): Complete words vocabulary for auto-correct
        tokenizer: Trained BPE tokenizer
        inverted_index (dict): Inverted index for search
        paragraph_tokens (dict): Paragraph token mappings
        paragraphs (list): All paragraph data
        threshold (int): Maximum edit distance for auto-correct
        
    Returns:
        list: Ranked search results
    """
    print(f"\nProcessing query: '{query}'")
    
    # Step 1: Split query into individual words
    query_words = query.lower().split()
    final_tokens = []
    
    print(f"Query words: {query_words}")
    
    # Step 2: Process each word individually
    for word in query_words:
        # Clean the word (remove non-alphabetic characters)
        word_clean = re.sub(r'[^a-z]', '', word)
        if not word_clean:
            continue
        
        print(f"\nProcessing word: '{word_clean}'")
        
        # Step 3: Check if word exists in vocabulary
        if word_clean in whole_words:
            # Word exists - tokenize directly with BPE
            print(f"  ✓ '{word_clean}' found in vocabulary")
            word_tokens = tokenizer.encode(word_clean)
            word_tokens = [t for t in word_tokens if t not in ['[UNK]', '[PAD]']]
            print(f"  Tokenizes to: {word_tokens}")
            final_tokens.extend(word_tokens)
            
        else:
            # Word not found - try auto-correct
            print(f"  ✗ '{word_clean}' NOT in vocabulary")
            
            # Step 4: Find correction suggestions using edit distance
            suggestions = suggest_corrections(word_clean, whole_words, max_dist=threshold, max_suggestions=5)
            
            if suggestions:
                # Present options to user
                print(f"  → Did you mean one of these?")
                for i, (suggestion, dist) in enumerate(suggestions, 1):
                    print(f"     {i}. {suggestion} (distance: {dist})")
                print(f"     0. Use original word '{word_clean}' (break into subwords)")
                
                # Get user input for choice
                while True:
                    try:
                        choice_input = input(f"  Enter your choice (0-{len(suggestions)}): ").strip()
                        choice = int(choice_input)
                        if 0 <= choice <= len(suggestions):
                            break
                        else:
                            print(f"  Please enter a number between 0 and {len(suggestions)}")
                    except ValueError:
                        print("  Please enter a valid number")
                
                if choice == 0:
                    # Use original word - break into BPE subwords
                    print(f"  → Using original word '{word_clean}'")
                    word_tokens = tokenizer.encode(word_clean)
                    word_tokens = [t for t in word_tokens if t not in ['[UNK]', '[PAD]']]
                    print(f"  Subwords: {word_tokens}")
                    final_tokens.extend(word_tokens)
                else:
                    # Use selected correction
                    selected_word, dist = suggestions[choice - 1]
                    print(f"  → Selected: {selected_word} (distance: {dist})")
                    corrected_tokens = tokenizer.encode(selected_word)
                    corrected_tokens = [t for t in corrected_tokens if t not in ['[UNK]', '[PAD]']]
                    print(f"  Corrected word tokenizes to: {corrected_tokens}")
                    final_tokens.extend(corrected_tokens)
            else:
                # No suggestions found - break into BPE subwords
                print(f"  → No correction found (min distance > {threshold})")
                print(f"  → Breaking into BPE subwords")
                word_tokens = tokenizer.encode(word_clean)
                word_tokens = [t for t in word_tokens if t not in ['[UNK]', '[PAD]']]
                print(f"  Subwords: {word_tokens}")
                final_tokens.extend(word_tokens)
    
    # Step 5: Execute search if we have tokens
    if not final_tokens:
        print("\nNo valid tokens found in query.")
        return []
    
    print(f"\nFinal query tokens: {final_tokens}")
    
    # Step 6: Perform search and return results
    results = search_query(final_tokens, inverted_index, paragraph_tokens, paragraphs)
    return results


def display_results(results):
    """
    Display search results in a formatted, user-friendly manner.
    
    Args:
        results (list): List of search result dictionaries
    """
    if not results:
        print("\nNo paragraphs found for this query.")
        return
    
    print(f"\n{'='*80}")
    print(f"Top {len(results)} Results:")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        # Create content snippet (first 200 characters)
        snippet = result['text'][:200].replace('\n', ' ')
        if len(result['text']) > 200:
            snippet += "..."
        
        # Display result with metadata
        print(f"\n[Rank {i}] Similarity Score: {result['score']:.4f}")
        print(f"  Paragraph ID: {result['id']}")
        print(f"  Book Title: {result['book']}")
        print(f"  Content Snippet: {snippet}")
        print(f"  {'-'*76}")


# ============================================================================
# PART 8: MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """
    Main function to run the search engine application.
    
    This function orchestrates the entire search engine setup and provides
    an interactive interface for users to search the corpus.
    """
    # Check for required dependencies
    if not TOKENIZERS_AVAILABLE:
        print("\nError: 'tokenizers' library is required.")
        print("Install it with: pip install tokenizers")
        return
    
    print("="*80)
    print("SEARCH ENGINE WITH AUTO-CORRECT")
    print("="*80)
    
    # Configuration
    corpus_dir = "/Users/myid/Documents/Projects/search/corpus"
    if not os.path.exists(corpus_dir):
        print(f"Error: Directory '{corpus_dir}' not found.")
        return
    
    # Load a sample of books for faster processing
    sample_n = 100
    # sample_n = 1000
    
    # Step 1: Load corpus
    print("\nLoading corpus...")
    paragraphs = load_corpus(corpus_dir, sample_n)
    if not paragraphs:
        print("No paragraphs loaded. Exiting.")
        return
    
    # Step 2: Train BPE tokenizer
    print("\nTraining BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=5000)
    tokenizer.fit([p['text'] for p in paragraphs])
    
    # Step 3: Build inverted index
    print("\nBuilding search index...")
    inverted_index, vocab, paragraph_tokens = build_inverted_index(paragraphs, tokenizer)
    
    # Step 4: Extract whole words for auto-correct
    print("Extracting whole words for auto-correct...")
    whole_words = extract_whole_words_from_corpus(paragraphs)
    print(f"Extracted {len(whole_words)} whole words for auto-correct.")
    
    # Step 5: Interactive search loop
    print("\n" + "="*80)
    print("Search engine ready! Enter your queries (type 'exit' or 'quit' to stop)")
    print("="*80)
    
    while True:
        query = input("\nEnter search query: ").strip()
        
        # Check for exit commands
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        # Skip empty queries
        if not query:
            print("Empty query. Please enter a search term.")
            continue
        
        # Process query and display results
        results = process_query(query, vocab, whole_words, tokenizer, inverted_index, paragraph_tokens, paragraphs)
        display_results(results)


if __name__ == "__main__":
    main()
