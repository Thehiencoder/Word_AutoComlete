import pickle
import tomotopy as tp
import spacy
import numpy as np
from Trie.trie import Trie, TrieNode


# ================== LOAD ==================

def load_models():
    lda_model = tp.LDAModel.load("LDA_CGS/lda_cgs.bin")

    word_to_id = {
        lda_model.used_vocabs[i]: i
        for i in range(lda_model.num_vocabs)
    }

    with open("Trie/Trie.pkl", 'rb') as f:
        trie = pickle.load(f)

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # (K, V)
    topic_word_matrix = np.array([
        lda_model.get_topic_word_dist(k)
        for k in range(lda_model.k)
    ])

    return lda_model, trie, word_to_id, nlp, topic_word_matrix


# ================== TOKENIZE ==================

def tokenize(text, nlp):
    doc = nlp(text.lower())
    return [
        t.lemma_ for t in doc
        if t.is_alpha
        and not t.is_stop
        and t.pos_ in ['NOUN', 'VERB', 'ADJ']
    ]


# ================== LDA ==================

def infer_topic_distribution(lda_model, context_tokens):
    if not context_tokens:
        return np.zeros(lda_model.k)

    doc = lda_model.make_doc(context_tokens)
    topic_dist, _ = lda_model.infer(doc)
    return np.array(topic_dist)


# ================== NORMALIZE (CACHE) ==================

normalize_cache = {}

def normalize_word_cached(word, nlp):
    if word in normalize_cache:
        return normalize_cache[word]

    doc = nlp(word.lower())
    if len(doc) == 0:
        normalize_cache[word] = word.lower()
        return normalize_cache[word]

    token = doc[0]
    lemma = token.lemma_ if token.is_alpha else word.lower()

    normalize_cache[word] = lemma
    return lemma


# ================== MAIN ==================

def suggest_words(
    lda_model,
    trie,
    word_to_id,
    nlp,
    topic_word_matrix,
    user_input,
    num_suggestions=5,
    verbose=False
):
    words_input = user_input.split()
    if not words_input:
        return []

    prefix = words_input[-1]
    context = " ".join(words_input[:-1]) if len(words_input) > 1 else ""

    if verbose:
        print(f"Context: '{context}'")
        print(f"Prefix: '{prefix}'")

    # ===== LDA context =====
    context_tokens = tokenize(context, nlp)
    context_topic_dist = infer_topic_distribution(lda_model, context_tokens)

    if context_topic_dist.sum() == 0:
        return []

    # ===== Trie candidates =====
    candidates = trie.topK(prefix, num_suggestions * 100)
    if not candidates:
        return []

    # ===== GROUP BY LEMMA =====
    lemma_groups = {}
    lemma_freq = {}

    for word, freq in candidates:
        lemma = normalize_word_cached(word, nlp)

        if lemma not in word_to_id:
            continue

        if lemma not in lemma_groups:
            lemma_groups[lemma] = []

        lemma_groups[lemma].append((word, freq))
        lemma_freq[lemma] = max(lemma_freq.get(lemma, 0), freq)

    if not lemma_groups:
        return []

    # ===== VECTORIZE =====
    lemmas = list(lemma_groups.keys())
    word_ids = np.array([word_to_id[l] for l in lemmas])

    # (K, N)
    word_topic_matrix = topic_word_matrix[:, word_ids]

    # Top topics
    top_k = 3
    top_idx = np.argsort(context_topic_dist)[-top_k:]

    context_top = context_topic_dist[top_idx]         # (k,)
    word_top = word_topic_matrix[top_idx, :]          # (k, N)

    # ===== SCORE =====
    scores = context_top @ word_top                   # (N,)

    # ===== SELECT BEST WORD PER LEMMA =====
    final_words = []
    final_scores = []

    for i, lemma in enumerate(lemmas):
        #Combine LDA + freq to improve the suggestion
        best_word, freq = max(lemma_groups[lemma], key=lambda x: x[1])

        freq_score = np.log(freq + 1)

        final_score = 0.9 * scores[i] + 0.2 * freq_score

        final_words.append(best_word)
        final_scores.append(final_score)

    final_scores = np.array(final_scores)

    # ===== SORT =====
    idx_sorted = np.argsort(final_scores)[::-1][:num_suggestions]

    result = [(final_words[i], final_scores[i]) for i in idx_sorted]

    if verbose:
        print("Top suggestions:", result)

    return result


# ================== INTERACTIVE ==================

def interactive_mode(lda_model, trie, word_to_id, nlp, topic_word_matrix):
    print("\n=== AUTOCOMPLETE MODE ===\n")

    while True:
        user_input = input("Nhập > ").strip()

        if user_input.lower() == "quit":
            break

        suggestions = suggest_words(
            lda_model,
            trie,
            word_to_id,
            nlp,
            topic_word_matrix,
            user_input,
            num_suggestions=10,
            verbose=True
        )

        print("\nGợi ý:")
        for i, (w, s) in enumerate(suggestions, 1):
            print(f"{i}. {w:20s} {s:.6f}")

        print()


# ================== MAIN ==================

def main():
    print("Loading...")
    lda_model, trie, word_to_id, nlp, topic_word_matrix = load_models()

    print(f"LDA topics: {lda_model.k}")
    print(f"Vocab size: {lda_model.num_vocabs}")

    interactive_mode(lda_model, trie, word_to_id, nlp, topic_word_matrix)


if __name__ == "__main__":
    main()