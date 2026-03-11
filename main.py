import streamlit as st
from transformers import pipeline
from PIL import Image

# タイトル
st.title("🐕 愛犬ごはんトッピングAI")

# 1. AIモデルの準備（キャッシュ機能で高速化）
@st.cache_resource
def load_model():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

detector = load_model()

# 2. 体重の入力
dog_weight = st.number_input("ワンちゃんの体重(kg)を入力してください", min_value=1.0, value=5.0)

# 3. 画像のアップロード
uploaded_file = st.file_uploader("食材の写真を撮るか、選んでください", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='アップロードされた画像', use_column_width=True)
    
    # 判定ボタン
    if st.button('この食材を判定する'):
        candidate_labels = ["sweet potato", "chicken breast", "broccoli", "onion", "chocolate"]
        res = detector(image, candidate_labels=candidate_labels)
        
        top_label = res[0]['label']
        st.success(f"AIの判定: 【{top_label}】")
        
        # ここから先の「トッピング計算」も st.write で表示していきます
