

# **Execution Instructions for Search Engine with Auto-Correct**

## Overview

This guide explains how to set up, run, and use the **Search Engine with Auto-Correct**. It covers system requirements, dependencies, file organization, execution methods, configuration, troubleshooting, and testing.

---

## 1. Prerequisites

### System Requirements

* **Python:** 3.7+
* **OS:** Windows, macOS, or Linux
* **RAM:** ≥ 4GB (8GB recommended for larger corpora)
* **Disk Space:** ≥ 1GB free

### Install Dependencies

```bash
pip install tokenizers
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

---

## 2. File Structure Setup

Organize files as follows:

```
search/
├── search_engine.py          # Main implementation
├── requirements.txt          # Python dependencies
├── corpus/                   # Directory containing text files
│   ├── book1.txt
│   ├── book2.txt
│   └── ...
└── README.md                 # Optional documentation
```

### Corpus Requirements

* Files in `.txt` format
* UTF-8 encoding
* Each file = one document
* Paragraphs separated by blank lines

Important: The corpus is not included in this repository (file size too large).
You need to create a new folder named corpus/ inside the project directory and place all your .txt book files there before running the search engine.
---

## 3. Execution Methods

### Method 1: Interactive Mode (Recommended)

Run:

```bash
python3 search_engine.py
```

Follow prompts:

1. Loads and processes corpus
2. Trains BPE tokenizer
3. Builds inverted index
4. Starts interactive search session

**Example:**

```
Enter search query: hapiness

→ Did you mean one of these?
  1. happiness (distance: 1)
  2. hapless (distance: 2)
  3. hardness (distance: 2)
  0. Use original word 'hapiness'
Enter choice (0–3): 1
```

Results returned with similarity ranking.

---

### Method 2: Programmatic Usage

Example usage inside Python:

```python
from search_engine import *

# Load corpus
paragraphs = load_corpus("corpus/", sample_n=100)

# Train BPE tokenizer
tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.fit([p['text'] for p in paragraphs])

# Build inverted index
inverted_index, vocab, para_tokens = build_inverted_index(paragraphs, tokenizer)

# Extract words for auto-correct
whole_words = extract_whole_words_from_corpus(paragraphs)

# Run search
results = process_query("hapiness", vocab, whole_words, tokenizer,
                        inverted_index, para_tokens, paragraphs)

display_results(results)
```

---

## 4. Configuration Options

* **Vocabulary size:** `BPETokenizer(vocab_size=5000)`
* **Auto-correct edit distance:** `threshold=2`
* **Number of results:** `top_k=10`

---

## 5. Troubleshooting

* **`tokenizers` not found:**

  ```bash
  pip install tokenizers
  ```
* **Corpus not loading:**

  * Ensure `.txt` files exist in `corpus/`
  * Check UTF-8 encoding
* **Memory errors:**

  * Reduce `vocab_size` (e.g., 2000)
  * Load fewer files with `sample_n`

---

## 6. Testing Queries

### Basic Queries

* `"beautiful"`
* `"love story"`

### Misspelled Queries (Auto-Correct)

* `"beutiful"` → `beautiful`
* `"hapiness"` → `happiness`

### Multi-word

* `"happy ending"`
* `"beautiful sunset"`

### Out-of-Vocabulary

* `"xyzabc"` → subword breakdown

---
