import sys
import os
import random
import spacy
import pickle
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Trie.trie import Trie, TrieNode 
from Trie_with_LDA.trie_with_lda import Trie_with_LDA, Trie_with_LDA_Node, load_models, suggest_words

#Load nlp once only
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
ALPHA_CONFIG = [0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5]

def load_test_data():
    path = "Dataset/raw_test_set.txt"
    if not os.path.exists(path): 
        print(f"Không tìm thấy file tại: {path}")
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Tách các bài báo dựa trên dấu phân cách @delimiter
        articles = content.split('@delimiter')
        
    # Loại bỏ các bài báo rỗng hoặc chỉ có dấu cách/xuống dòng
    cleaned_articles = [a.strip() for a in articles if len(a.strip()) > 20]
    return cleaned_articles

def pre_tokenize_articles(articles):
    tokenized_data = []
    print(f"Đang xử lý {len(articles)} bài báo...")
    
    # nlp.pipe giúp xử lý nhanh hơn rất nhiều so với vòng lặp thường
    for doc in nlp.pipe(articles, batch_size=512, disable=['parser', 'ner']):
        # Làm sạch:
        # - Chuyển về chữ thường (lowercase)
        # - Loại bỏ dấu câu (is_punct: -- , . ! ?)
        # - Loại bỏ khoảng trắng/xuống dòng (is_space)
        # - Loại bỏ các ký tự đặc biệt không phải chữ/số (is_alpha) nếu cần
        tokens = [token.lemma_ for token in doc 
                  if token.is_alpha and not token.is_punct and not token.is_space and not token.text == "--" and not token.is_stop]
        
        if tokens:
            tokenized_data.append(tokens)
            
    return tokenized_data

def evaluate_hit_at_k(method_name, tokenized_articles, trie_obj, lda_params=None, k=10, check_matrix=None):
    hit_count = 0
    total_queries = 0
    max_prefix_len = 6
    hit_counts_by_prefix = {l: 0 for l in range(1, max_prefix_len + 1)}
    total_queries_by_prefix = {l: 0 for l in range(1, max_prefix_len + 1)}

    #Extract LDA
    lda_m, w2id, tw_matrix = lda_params[:3] if lda_params else (None, None, None)

    for doc_i, tokens in enumerate(tokenized_articles):
        if not tokens: continue
        for wi in check_matrix[doc_i]:
            if wi >= len(tokens): continue
            word = tokens[wi]
            # Context window: 20 words
            context_str = " ".join(tokens[max(0, wi-20):wi])

            for prefix_len in range(1, min(len(word), max_prefix_len) + 1):
                prefix = word[:prefix_len]
                
                if lda_params:
                    #Add nlp global into suggest_words
                    suggestions = suggest_words(trie_obj, lda_m, w2id, tw_matrix, 
                                               nlp, f"{context_str} {prefix}".strip(), K=k, alpha=ALPHA_CONFIG)
                    suggested_words = [w for w, _ in suggestions]
                else:
                    suggested_words = [w for w, _ in trie_obj.topK(prefix, k)]
                 
                print(f"--- Testing Method: {method_name} ---")
                print(f"Target: {word} | Prefix: {prefix}")
                print(f"Top {k} Suggestions: {suggested_words}")
                if word in suggested_words[:k]:
                    hit_count += 1
                    hit_counts_by_prefix[prefix_len] += 1
                total_queries += 1
                total_queries_by_prefix[prefix_len] += 1

    return {
        "k": k, "method": method_name,
        "hit_rates_by_prefix": {pl: (hit_counts_by_prefix[pl] / total_queries_by_prefix[pl] if total_queries_by_prefix[pl] > 0 else 0) 
                                for pl in range(1, max_prefix_len + 1)}
    }

if __name__ == "__main__":
    articles = load_test_data()[:100]
    tokenized_articles = pre_tokenize_articles(articles)
    
    #Check matrix dùng chung để đảm bảo tính công bằng (fair comparison)
    check_matrix = [random.sample(range(len(t)), min(len(t), 10)) for t in tokenized_articles]
    print(f"Số lượng bài báo load được: {len(tokenized_articles)}")
    print(f"Số lượng vị trí test trong bài 1: {len(check_matrix[0]) if check_matrix else 0}")
    lda_info = load_models()
    
    sys.modules['__main__'].TrieNode = TrieNode
    sys.modules['__main__'].Trie = Trie
    sys.modules['__main__'].Trie_with_LDA_Node = Trie_with_LDA_Node
    sys.modules['__main__'].Trie_with_LDA = Trie_with_LDA

    with open("Trie/Trie.pkl", 'rb') as f: trie_only = pickle.load(f)
    with open("Trie_with_LDA/Trie_with_LDA.pkl", 'rb') as f: trie_lda = pickle.load(f)

    res_trie = evaluate_hit_at_k("Trie Only", tokenized_articles, trie_only, check_matrix=check_matrix)
    res_lda = evaluate_hit_at_k("Trie + LDA", tokenized_articles, trie_lda, lda_params=lda_info, check_matrix=check_matrix)

    results = []
    for pl in range(1, 7):
        results.append({
            "Prefix": pl,
            "Trie Only": f"{res_trie['hit_rates_by_prefix'][pl]:.4f}",
            "Trie + LDA": f"{res_lda['hit_rates_by_prefix'][pl]:.4f}"
        })
    print(pd.DataFrame(results).to_string(index=False))
