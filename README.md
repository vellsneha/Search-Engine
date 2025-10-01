

# 🔍 Search Engine with Auto-Correct

A lightweight search engine that combines **Byte Pair Encoding (BPE) tokenization** and an **edit-distance-based auto-correct system** to handle misspellings and out-of-vocabulary words. Built as part of a course project—fun to build and a great way to explore how search engines work!


## ✨ Features

* **BPE Tokenization** for subword-level search
* **Auto-Correct** using edit distance
* **Cosine Similarity Ranking** for relevance
* **Handles OOV words** gracefully


## ⚙️ Installation

```bash
git clone https://github.com/vellsneha/Search-Engine.git
cd search
pip install -r requirements.txt
```


## 🚀 Usage

Run directly:

```bash
python3 search_engine.py
```

Or use programmatically:

```python
from search_engine import *

paragraphs = load_corpus("path/to/corpus", sample_n=100)
tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.fit([p['text'] for p in paragraphs])
inverted_index, vocab, paragraph_tokens = build_inverted_index(paragraphs, tokenizer)

results = process_query("beutiful", vocab, extract_whole_words_from_corpus(paragraphs), 
                        tokenizer, inverted_index, paragraph_tokens, paragraphs)
display_results(results)
```


## ⚡ How It Works

### 🔡 Byte Pair Encoding (BPE)

BPE is a **subword tokenization** method. Instead of splitting text only into whole words, it learns frequent character pairs and merges them into subwords.

* Example: `"happiness"` → `["hap", "pi", "ness"]`
* This helps the search engine handle **rare or unseen words** by breaking them into meaningful sub-units.
* In this project: I used BPE to tokenize both the corpus and the queries, which makes matching more robust.

### ✏️ Edit Distance (Levenshtein Distance)

Edit distance measures how many operations (insert, delete, substitute) are needed to transform one word into another.

* Example: `"beutiful"` → `"beautiful"` has distance 1 (swap “u” → “a”).
* In this project: If a query word isn’t in the vocabulary, I compare it to all known words and suggest the **closest matches**.

👉 Together, **BPE handles unknown words**, and **edit distance fixes misspellings**, making search results both flexible and accurate.


## 📝 Example Queries

* `beutiful` → Suggests **beautiful**
* `hapiness` → Suggests **happiness**
* `sunsett` → Suggests **sunset**


## 📚 Project Note

This was a **course project**—a hands-on dive into **tokenization, auto-correct algorithms, and information retrieval**. I really enjoyed building it and learned a lot about search systems.


## 📬 Contact

Feel free to open an issue if you’d like to discuss improvements!

