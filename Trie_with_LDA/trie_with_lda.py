import heapq
import pickle
import spacy
import tomotopy as tp
import numpy as np


def load_models():
    lda_model = tp.LDAModel.load("LDA_CGS/lda_cgs.bin")

    word_to_id = {
        lda_model.used_vocabs[i]: i
        for i in range(lda_model.num_vocabs)
    }

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    topic_word_matrix = np.array([
        lda_model.get_topic_word_dist(k)
        for k in range(lda_model.k)
    ])

    return lda_model, word_to_id, topic_word_matrix, nlp

_tokenize_cache = {}

def tokenize(text, nlp):
    key = text.strip().lower()
    if not key:
        return ""

    if key in _tokenize_cache:
        return _tokenize_cache[key]

    doc = nlp(key)
    lemmatized_token = ""
    for t in doc:
        if (
            t.is_alpha and
            not t.is_punct and
            not t.is_space and
            not t.is_stop and
            t.pos_ in ['NOUN', 'VERB', 'ADJ']
        ):
            lemmatized_token = t.lemma_
            break

    _tokenize_cache[key] = lemmatized_token
    return lemmatized_token


class Trie_with_LDA_Node:
    def __init__(self):
        self.child = {}
        self.is_end = False
        self.freq = 0
        self.topic_dist = np.zeros(170)


class Trie_with_LDA:
    def __init__(self):
        self.root = Trie_with_LDA_Node()
        self.topic_context_dist = None

    def insert(self, word, word_to_id, topic_word_matrix, nlp):
        cur = self.root
        for c in word:
            if c not in cur.child:
                cur.child[c] = Trie_with_LDA_Node()
            cur = cur.child[c]

        cur.is_end = True
        cur.freq += 1
        
        if cur.topic_dist.sum() == 0:
            word = tokenize(word, nlp)

            if  word in word_to_id:
                word_id = word_to_id[word]
                cur.topic_dist = topic_word_matrix[:, word_id]

    def infer_topic_dist(self, lda_model, word_to_id, topic_word_matrix, context, nlp):
        context_tokens = [t for word in context.split() if (t := tokenize(word, nlp))][-20:]

        if not context_tokens:
            self.topic_context_dist = np.zeros(lda_model.k)
        else:
            phi_vectors = []
            for token in context_tokens:
                if token in word_to_id:
                    word_id = word_to_id[token]
                    phi_vectors.append(topic_word_matrix[:, word_id])
            
            if phi_vectors:
                topic_sum = np.sum(phi_vectors, axis=0)
                self.topic_context_dist = topic_sum / np.sum(topic_sum)
            else:
                self.topic_context_dist = np.zeros(lda_model.k)

    def _dfs(self, node : Trie_with_LDA_Node, cur_word, K, heap, alpha, max_freq, len_prefix):
        if node.is_end:
            # Normalize freq to [0, 1]
            norm_freq = np.log(1 + node.freq) / np.log(1 + max_freq)
            
            # Normalize dot product (cosine similarity)
            node_norm = np.linalg.norm(node.topic_dist)
            context_norm = np.linalg.norm(self.topic_context_dist)
            if node_norm > 0 and context_norm > 0:
                norm_similarity = (node.topic_dist @ self.topic_context_dist) / (node_norm * context_norm)
            else:
                norm_similarity = 0
            
            # Combine freq and similarity with alpha (Heuristic)
            score = norm_freq * (1 + alpha[len_prefix] * norm_similarity ** 2)
            if len(heap) < K:
                heapq.heappush(heap, (score, cur_word))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, cur_word))

        for c, nxt in node.child.items():
            self._dfs(nxt, cur_word + c, K, heap, alpha, max_freq, len_prefix)

    def topK(self, prefix, K, alpha):
        cur = self.root
        for c in prefix:
            if c not in cur.child:
                return []
            cur = cur.child[c]

        # Find max frequency in subtree for normalization
        def find_max_freq(node):
            max_f = node.freq if node.is_end else 0
            for child in node.child.values():
                max_f = max(max_f, find_max_freq(child))
            return max_f
        
        max_freq = find_max_freq(cur)
        
        heap = []
        self._dfs(cur, prefix, K, heap, alpha, max_freq, len_prefix=len(prefix))

        return sorted(
            [(word, score) for score, word in heap],
            key=lambda x: x[1],
            reverse=True
        )


def build_trie_with_lda(word_to_id, topic_word_matrix, nlp):
    trie = Trie_with_LDA()

    with open("Dataset/training_data_for_Trie.pkl", 'rb') as f:
        tokenized_articles = pickle.load(f)

    for i, doc in enumerate(tokenized_articles):
        print(f"Processing document {i+1}/{len(tokenized_articles)}")
        for word in doc:
            trie.insert(word, word_to_id, topic_word_matrix, nlp)

    return trie


def suggest_words(trie_with_lda : Trie_with_LDA, lda_model, word_to_id, topic_word_matrix, nlp, user_input, K, alpha):
    words_input = user_input.split()
    if not words_input:
        return []

    prefix = words_input[-1]
    context = " ".join(words_input[:-1]) if len(words_input) > 1 else ""

    trie_with_lda.infer_topic_dist(lda_model, word_to_id, topic_word_matrix, context, nlp)
    topK = trie_with_lda.topK(prefix, K, alpha)

    return topK


if __name__ == "__main__":
    lda_model, word_to_id, topic_word_matrix, nlp = load_models()
    # trie_with_lda = build_trie_with_lda(word_to_id, topic_word_matrix, nlp)

    # Save
    # filename = "Trie_with_LDA/Trie_with_LDA.pkl"
    # with open(filename, 'wb') as f:
    #     pickle.dump(trie_with_lda, f)

    # Load
    with open("Trie_with_LDA/Trie_with_LDA.pkl", 'rb') as f:
        trie_with_lda = pickle.load(f)

    print("Models loaded. Ready for suggestions.")
    K = 10
    user_input = "machine learning is very po"
    alpha = [None, 0.25, 1.75, 2.5, 2.0, 1.75, 0.0]
    topK = suggest_words(trie_with_lda, lda_model, word_to_id, topic_word_matrix, nlp, user_input, K, alpha)
    [print(word, score) for word, score in topK]