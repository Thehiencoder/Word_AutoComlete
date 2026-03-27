import streamlit as st
import pickle
import os
import sys
import time
import spacy
import tomotopy as tp
import numpy as np
from st_keyup import st_keyup
import gdown

#Realize custom module in project dir
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

try:
    from Trie_with_LDA.trie_with_lda import Trie_with_LDA, Trie_with_LDA_Node, load_models, suggest_words
except ImportError:
    st.error("Không tìm thấy file logic 'trie_with_lda.py'. Vui lòng kiểm tra lại cấu trúc thư mục!")

#Setting
st.set_page_config(
    page_title="Context-Aware Auto-Suggestion System",
    layout="centered"
)

st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #4CAF50;
        color: #4CAF50;
        background-color: transparent;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

#Load resource
@st.cache_resource
def load_all_assets():

    trie_dir = os.path.join(CURRENT_DIR, "Trie_with_LDA")
    lda_dir = os.path.join(CURRENT_DIR, "LDA_CGS") # Folder containing .bin file

    os.makedirs(trie_dir, exist_ok=True)
    os.makedirs(lda_dir, exist_ok=True)

    FILES_TO_DOWNLOAD = {
        os.path.join(trie_dir, "Trie_with_LDA.pkl"): "1LjP1jyfpQqOIHtqpfzJGYYg4PvLh-TJV",
        os.path.join(lda_dir, "lda_cgs.bin"): "1wB88ZNGPtYOInZqupomViQN1d3nym-kh"
    }

    for destination, file_id in FILES_TO_DOWNLOAD.items():
        if not os.path.exists(destination):
            with st.spinner(f"Đang tải {os.path.basename(destination)}..."):
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, destination, quiet=False)
    
    status = st.empty()

    status.info("Đang khởi tạo hệ thống LDA & Trie... Vui lòng đợi trong giây lát.")
    
    try:
        #Load needed components
        lda_m, w2id, tw_matrix, nlp = load_models()
        
        #Load Trie Object
        trie_path = os.path.join(CURRENT_DIR, "Trie_with_LDA", "Trie_with_LDA.pkl")
        if not os.path.exists(trie_path):
            st.error(f"Không tìm thấy file: {trie_path}")
            return None
            
        with open(trie_path, 'rb') as f:
            trie_obj = pickle.load(f)
            
        status.success("Hệ thống đã sẵn sàng!")
        time.sleep(1) 
        status.empty()
        
        return lda_m, w2id, tw_matrix, nlp, trie_obj
    except Exception as e:
        status.error(f"Lỗi khi nạp Model: {str(e)}")
        return None

assets = load_all_assets()
if assets:
    lda_m, w2id, tw_matrix, nlp, trie_obj = assets
else:
    st.stop() #Stop the app if failed to load model

ALPHA_VALS = [None, 0.25, 1.75, 2.5, 2.0, 1.75, 0.0]

#Manage the current app status
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

if 'input_widget_key' not in st.session_state:
    st.session_state.input_widget_key = 0

def handle_selection(full_word):
    #Add chosen word into input
    parts = st.session_state.input_text.rsplit(" ", 1)
    if len(parts) > 1:
        new_text = parts[0] + " " + full_word + " "
    else:
        new_text = full_word + " "
    st.session_state.input_text = new_text
    st.session_state.input_widget_key += 1

st.title("Context-Aware Auto-Suggestion")
st.markdown("Hệ thống gợi ý từ thông minh kết hợp Trie Top-K và Ngữ cảnh LDA.")

user_input = st_keyup(
    label="Nhập văn bản của bạn:",
    placeholder="Ví dụ: machine learning is very po...",
    key=f"keyup_field_{st.session_state.input_widget_key}",
    value=st.session_state.input_text,
    debounce=250, #Wait for 250ms after stop typing to process
    label_visibility="collapsed"
)

#Update session_state to synchronized with user_input
st.session_state.input_text = user_input

if user_input:
    
    prefix = user_input.split(" ")[-1].lower()
    
    if prefix:
        try:
            #Call word suggestion function
            raw_suggestions = suggest_words(
                trie_obj, lda_m, w2id, tw_matrix, nlp, 
                user_input, K=5, alpha=ALPHA_VALS
            )
            
            suggestions = [s[0] for s in raw_suggestions]
            
            if suggestions:
                st.caption(f"Gợi ý cho '**{prefix}**':")
                cols = st.columns(len(suggestions))
                for i, word in enumerate(suggestions):
                    cols[i].button(
                        word, 
                        key=f"btn_{word}_{i}", 
                        on_click=handle_selection, 
                        args=(word,)
                    )
        except Exception as e:
            st.write("...") 

with st.sidebar:
    st.header("Thông số Model")
    st.divider()
    st.write(f"**LDA Model:** {lda_m.k} Topics")
    st.write(f"**Từ điển:** {lda_m.num_vocabs:,} từ")
    st.write(f"**Trạng thái:** Đang chạy (Cached)")
    
    if st.button("Xóa toàn bộ văn bản", use_container_width=True):
        st.session_state.input_text = ""
        st.session_state.input_widget_key += 1
        st.rerun()
    
    st.divider()
    
    
