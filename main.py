import streamlit as st
from transformers import pipeline
from PIL import Image

# ページ全体のデザイン設定
st.set_page_config(
    page_title="DogFood AI", 
    page_icon="🐶", 
    layout="centered"
)

# カスタムCSSでデザインを微調整（背景色やフォントなど）
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .result-card {
        padding: 20px;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🐶 愛犬ごはんコンシェルジュ")
st.caption("写真を撮るだけで、今日のご飯の適量をアドバイスします")

# AIモデルの準備
@st.cache_resource
def load_model():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")

detector = load_model()

# --- メインレイアウト ---
# 2つのカラムに分ける
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 情報を入力")
    dog_weight = st.number_input("ワンちゃんの体重(kg)", min_value=0.1, value=5.0, step=0.1)
    uploaded_file = st.file_uploader("食材の写真をアップロード", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("2. 判定結果")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        if st.button('AI判定スタート！'):
            with st.spinner('計算中...'):
                candidate_labels = ["sweet potato", "chicken breast", "broccoli", "onion", "chocolate", "apple"]
                res = detector(image, candidate_labels=candidate_labels)
                top_label = res[0]['label']
                
                # 食材データ
                food_info = {
                    "sweet potato": {"name": "さつまいも", "kcal": 130, "icon": "🍠"},
                    "chicken breast": {"name": "ささみ", "kcal": 105, "icon": "🍗"},
                    "broccoli": {"name": "ブロッコリー", "kcal": 35, "icon": "🥦"},
                    "apple": {"name": "りんご", "kcal": 50, "icon": "🍎"},
                    "onion": {"name": "たまねぎ", "safe": False},
                    "chocolate": {"name": "チョコ", "safe": False}
                }

                st.markdown("---")
                if top_label in food_info:
                    info = food_info[top_label]
                    if info.get("safe", True):
                        # 適量計算
                        daily_limit_kcal = dog_weight * 70 * 0.1
                        max_grams = (daily_limit_kcal / info["kcal"]) * 100
                        reduce_food = daily_limit_kcal / 3.5
                        
                        # カード風の表示
                        st.success(f"判定：{info['icon']} {info['name']}")
                        st.metric("今日のトッピング上限", f"{max_grams:.1f}g")
                        st.info(f"💡 ドッグフードを約{reduce_food:.1f}g減らして調整しましょう！")
                    else:
                        st.error(f"🚨 警告：{info['name']}は犬に与えてはいけません！")
    else:
        st.write("写真をアップロードするとここに結果が表示されます")

st.divider()
st.info("※このアプリは目安を表示するものです。体調に合わせて調整してください。")
