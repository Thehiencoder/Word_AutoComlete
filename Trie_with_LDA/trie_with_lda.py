import heapq
import pickle
import spacy
import tomotopy as tp
import numpy as np
from functools import lru_cache

def load_models():
    #Load the pre-trained LDA model
    lda_model = tp.LDAModel.load("LDA_CGS/lda_cgs.bin")
    #Map words to their internal IDs in the LDA model for quick lookup
    word_to_id = {
        lda_model.used_vocabs[i]: i
        for i in range(lda_model.num_vocabs)
    }
    #Extract the full topic-word distribution matrix (shape: K_topics x V_vocab)
    topic_word_matrix = np.array([
        lda_model.get_topic_word_dist(k)
        for k in range(lda_model.k)
    ], dtype=np.float32)

    nlp_model = spacy.load('en_core_web_sm', disable=['parser','ner'])

    return lda_model, word_to_id, topic_word_matrix, nlp_model


@lru_cache(maxsize=20000)
def tokenize(text, nlp_ref): #Using cache to avoid calling Spacy repeatedly.
    #Preprocess text using lowercase, lemmatization, removing space, stopwords, and filtering for meaningful POS tags
    doc = nlp_ref(text.lower().strip())
    for t in doc:
        if t.is_alpha and not t.is_punct and not t.is_space and not t.is_stop and t.pos_ in ['NOUN', 'VERB', 'ADJ']:
            return t.lemma_.lower()
    return ""


class Trie_with_LDA_Node:
    def __init__(self):
        self.child = {}
        self.is_end = False
        self.freq = 0
        self.max_subtree_freq = 0 #topK doesn't have to traverse the tree to find max
        self.topic_dist = None #Stores the P(word|topic) vector for this specific word


class Trie_with_LDA:
    def __init__(self, k_topics = 20):
        self.root = Trie_with_LDA_Node()
        self.topic_context_dist = np.zeros(k_topics) #Global placeholder for current context inference

    def insert(self, word, word_to_id, topic_word_matrix, nlp):
        cur = self.root
        nodes_on_path = [cur]
        word_lower = word.lower()
        for c in word_lower:
            if c not in cur.child:
                cur.child[c] = Trie_with_LDA_Node()
            cur = cur.child[c]
            nodes_on_path.append(cur)

        cur.is_end = True
        cur.freq += 1
        
        #Update max_subtree_freq back to parent nodes
        for node in nodes_on_path:
            if cur.freq > node.max_subtree_freq:
                node.max_subtree_freq = cur.freq

        #Assign LDA vector to word node
        if cur.topic_dist is None:
            
            if word_lower in word_to_id:
                word_id = word_to_id[word_lower]
                cur.topic_dist = topic_word_matrix[:, word_id]

    def infer_topic_dist(self, lda_model, word_to_id, topic_word_matrix, context, nlp):
        words = context.split()
        context_tokens = []
        for w in words[-20:]: #Look for last 20 words for context relevance
            t = tokenize(w, nlp)
            if t: context_tokens.append(t)

        if not context_tokens:
            self.topic_context_dist = np.zeros(lda_model.k)
            return 
        #Collect topic vectors for all valid tokens in the context
        phi_vectors = [topic_word_matrix[:, word_id] 
                       for t in context_tokens 
                       if (word_id := word_to_id.get(t)) is not None]
        
        if phi_vectors:
            #Average/Sum the vectors to get a single representative context vector
            topic_sum = np.sum(phi_vectors, axis=0)
            self.topic_context_dist = topic_sum / (np.sum(topic_sum) + 1e-9)
        else:
            self.topic_context_dist = np.zeros(lda_model.k)

    def _dfs(self, node, cur_word, K, heap, alpha_val, max_freq):
        if node.is_end:
            #Log-normalized global frequency
            norm_freq = np.log(1 + node.freq) / np.log(1 + max_freq) if max_freq > 0 else 0

            #Cosine similarity between word topic and context topic
            norm_similarity = 0
            if node.topic_dist is not None:
                node_norm = np.linalg.norm(node.topic_dist)
                context_norm = np.linalg.norm(self.topic_context_dist)
                if node_norm > 0 and context_norm > 0:
                    norm_similarity = (node.topic_dist @ self.topic_context_dist) / (node_norm * context_norm)
            
            #Calculate final score
            #Alpha controls the weight of the context influence.
            score = norm_freq * (1 + alpha_val * (norm_similarity ** 2))
            #score = norm_freq + alpha_val * norm_similarity
            
            if len(heap) < K:
                heapq.heappush(heap, (score, cur_word))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, cur_word))

        for c, nxt in node.child.items():
            self._dfs(nxt, cur_word + c, K, heap, alpha_val, max_freq)

    def topK(self, prefix, K, alpha_list):
        cur = self.root
        for c in prefix:
            if c not in cur.child:
                return []
            cur = cur.child[c]

        #Take alpha based on prefix length
        alpha_idx = min(len(prefix), len(alpha_list) - 1)
        alpha_val = alpha_list[alpha_idx] if alpha_list[alpha_idx] is not None else 0
        
        heap = []
        
        self._dfs(cur, prefix, K, heap, alpha_val, cur.max_subtree_freq)
        #Return results sorted by score in descending order
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


def suggest_words(trie_with_lda, lda_model, word_to_id, topic_word_matrix, nlp, user_input, K, alpha):
    #Process raw user input and return context-aware suggestions.
    words_input = user_input.split()
    if not words_input: return []

    prefix = words_input[-1].lower()
    context = " ".join(words_input[:-1])

    trie_with_lda.infer_topic_dist(lda_model, word_to_id, topic_word_matrix, context, nlp)
    return trie_with_lda.topK(prefix, K, alpha)


if __name__ == "__main__":
    lda_model, word_to_id, topic_word_matrix, nlp = load_models()
    trie_with_lda = build_trie_with_lda(word_to_id, topic_word_matrix, nlp)

    # Save
    filename = "Trie_with_LDA/Trie_with_LDA.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(trie_with_lda, f)

    # Load
    with open("Trie_with_LDA/Trie_with_LDA.pkl", 'rb') as f:
        trie_with_lda = pickle.load(f)

    print("Models loaded. Ready for suggestions.")
    K = 10
    user_input = "machine learning is very po"
    alpha = [None, 0.25, 1.75, 2.5, 2.0, 1.75, 0.0]
    topK = suggest_words(trie_with_lda, lda_model, word_to_id, topic_word_matrix, nlp, user_input, K, alpha)
    [print(word, score) for word, score in topK]
