import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="愛犬ごはんトッピングAI", page_icon="🐶")
st.title("🐕 愛犬ごはんトッピングAI")

# 1. モデルのロード（メモリ節約のため、軽量な patch16 を使用）
@st.cache_resource
def load_model():
    # 元の patch32 よりも少し精度が良く、メモリ効率も調整されたモデルに変更
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")

with st.spinner('AIの準備中...（初回は数分かかります）'):
    detector = load_model()

# 2. 基本情報入力
dog_weight = st.number_input("ワンちゃんの体重(kg)を入力してください", min_value=0.1, value=5.0, step=0.1)
daily_limit_kcal = dog_weight * 70 * 0.1

# 3. 画像アップロード
uploaded_file = st.file_uploader("食材の写真を撮るか選んでください", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='判定する画像', use_container_width=True)
    
    # 判定ボタン
    if st.button('この食材を判定する'):
        with st.spinner('AIが考え中...'):
            candidate_labels = ["sweet potato", "chicken breast", "broccoli", "onion", "chocolate", "apple"]
            res = detector(image, candidate_labels=candidate_labels)
            
            top_label = res[0]['label']
            score = res[0]['score']
            
            st.success(f"AIの判定: 【{top_label}】 (確信度: {score:.2f})")
            
            # 簡易的なトッピング計算表示
            kcal_db = {"sweet potato": 130, "chicken breast": 105, "broccoli": 35, "apple": 50}
            
            if top_label in ["onion", "chocolate"]:
                st.error("⚠️ 警告: 犬にとって非常に危険です！")
            elif top_label in kcal_db:
                max_grams = (daily_limit_kcal / kcal_db[top_label]) * 100
                st.info(f"💡 体重{dog_weight}kgの場合、1日最大 {max_grams:.1f}g までが目安です。")
