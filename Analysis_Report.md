# Search Engine with Auto-Correct: Detailed Analysis

## 1. Five Sample Queries and Search Results

I tested 5 different query types to analyze how the system responds:

### Query 1: **"beautiful"** (Correct spelling, word exists)

* **Result:** 10 highly relevant matches (e.g., “Beautiful,” said Stella…)
* **Why retrieved:** Exact match found in vocabulary and documents.
* **Auto-correct:** Not needed.
* **Errors:** None.

---

### Query 2: **"hapiness"** (Misspelling)

* **Result:** 10 matches for *happiness* (e.g., “To study their happiness all the way up to town…”)
* **Why retrieved:** Auto-correct suggested *happiness* (edit distance 1). After correction, BPE tokenization allowed a good match.
* **Auto-correct:** Critical — without it, no results would appear.
* **Errors:** None, correction worked perfectly.

---

### Query 3: **"love story"** (Multi-word query)

* **Result:** 10 matches mentioning love and stories.
* **Why retrieved:** Both “love” and “story” existed in the vocabulary; BPE tokenized *story* into `['sto', 'ry']` but still matched.
* **Auto-correct:** Not needed.
* **Errors:** Slight semantic looseness (sometimes “story” matches contexts not strictly about love stories).

---

### Query 4: **"xyzabc"** (Nonsense / Out-of-Vocabulary)

* **Result:** 10 partial matches on subwords (like “x”, “ab”).
* **Why retrieved:** BPE broke the word into smaller tokens that overlapped with existing vocabulary.
* **Auto-correct:** Could not help — no close candidates within distance 2.
* **Errors:** No meaningful retrieval — results were noise, but system degraded gracefully instead of failing.

---

### Query 5: **"sunsett"** (Misspelling)

* **Result:** 10 matches about *sunset* (e.g., poetry mentioning sunset).
* **Why retrieved:** Auto-correct suggested *sunset* (edit distance 1). Tokenization → `['sun', 'set']`.
* **Auto-correct:** Crucial — without it, no matches.
* **Errors:** None, correct fix applied.

---

**Summary of Queries:**

* Auto-correct **saved queries 2 & 5**.
* Queries 1 & 3 worked directly.
* Query 4 highlighted robustness — no crash, but retrieval was weak.

---

## 2. Effect of BPE vs Whitespace Tokenization

I compared results using **BPE** and **whitespace tokenization** with the query:

**Query:** `"beautiful sunset"`

* **BPE tokens:** `['beaut', 'if', 'ul', 'sun', 'set']`
* **Whitespace tokens:** `['beautiful', 'sunset']`

### Impact on Results:

* **BPE:**

  * Handles OOV gracefully (can still tokenize unknowns).
  * Finds partial matches (e.g., documents with “beauty” + “setting sun”).
  * Smaller vocabulary (~1,000 tokens).

* **Whitespace:**

  * Requires exact matches (fails if typo or variation).
  * Larger vocabulary (~24,000 tokens).
  * Cleaner semantic matches (keeps words intact).

**Conclusion:** Without BPE, results are brittle: typos like *sunsett* or *hapiness* would return **zero hits**, whereas BPE enables correction and fallback via subwords.

---

## 3. Effect of Edit Distance Threshold on Auto-Correct

Test word: **"hapiness"**

* **Threshold = 1:** Suggests only *happiness*.

  * High precision, but misses other possibilities.
  * Works well for this query.

* **Threshold = 2:** Suggests *happiness, hapless, harness, hardness, holiness*.

  * Balanced precision/recall.
  * User may need to choose.

* **Threshold = 3:** Same 5 suggestions as threshold 2.

  * No real gain, only adds computational cost.

**Impact on Results:**

* Low threshold → fewer suggestions, risk of missing intended correction.
* High threshold → more suggestions, but risk of irrelevant noise.
* **Optimal threshold = 2** → catches most real-world typos while avoiding overwhelming the user.

---

## 4. Final Insights

1. **Sample Queries:** Demonstrated exact matches, typos, multi-word queries, and nonsense inputs. Auto-correct proved crucial for user satisfaction.
2. **BPE vs Whitespace:** BPE provides robustness and partial matching, while whitespace fails hard on typos and OOV words.
3. **Edit Distance:** Threshold 2 is optimal — balances recall and precision.

**Overall:** The system consistently provides meaningful results, handles user errors gracefully, and demonstrates the strength of combining **auto-correct, BPE tokenization, and edit-distance correction** with a traditional inverted index.

