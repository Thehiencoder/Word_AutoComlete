import pickle

class BasicTrieNode:
    def __init__(self):
        self.child = {}
        self.is_end = False
        self.freq = 0

class BasicTrie:
    def __init__(self):
        self.root = BasicTrieNode()

    def insert(self, word):
        word = word.lower()
        cur = self.root
        for c in word:
            if c not in cur.child:
                cur.child[c] = BasicTrieNode()
            cur = cur.child[c]
        cur.is_end = True

    def _dfs(self, node, cur_word, K, results):
        #Stop when getting K results
        if len(results) >= K:
            return

        if node.is_end:
            results.append(cur_word)

        #Traverse chars in alphabetical order
        for c in sorted(node.child.keys()):
            if len(results) >= K:
                break
            self._dfs(node.child[c], cur_word + c, K, results)

    def topK(self, prefix, K):
        prefix = prefix.lower()
        cur = self.root
        for c in prefix:
            if c not in cur.child:
                return []
            cur = cur.child[c]

        results = []
        self._dfs(cur, prefix, K, results)
        
        #Return word list, freq=0 for all words (as we are not using freq)
        return [(word, 0) for word in results]

def build_basic_trie():
    trie = BasicTrie()
    with open("Dataset/training_data_for_Trie.pkl", 'rb') as f:
        tokenized_articles = pickle.load(f)
    
    #Insert words into trie
    for doc in tokenized_articles:
        for w in doc:
            trie.insert(w)
    return trie

if __name__ == "__main__":
    basic_trie = build_basic_trie()
    #Save
    with open("Trie/Trie_Basic.pkl", 'wb') as f:
        pickle.dump(basic_trie, f)

    #Load
    with open("Trie/Trie_Basic.pkl", 'rb') as f:
         trie = pickle.load(f)
    print(trie.topK("app", 5))