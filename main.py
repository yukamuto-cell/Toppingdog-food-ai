import streamlit as st
from transformers import pipeline
from PIL import Image

# ページ設定
st.set_page_config(page_title="DogFood AI", page_icon="🐶")

# デザイン調整
st.markdown("""
    <style>
    .stButton>button { border-radius: 20px; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🐶 愛犬トッピング・チェッカー")

# AIモデルの準備
@st.cache_resource
def load_model():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")

detector = load_model()

# --- 入力エリア ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. ワンちゃんの情報")
    dog_weight = st.number_input("体重(kg)", min_value=0.1, value=5.0, step=0.1)
    # ユーザーが使いたい量を入力
    use_grams = st.number_input("与えたい分量(g)", min_value=1, value=10, step=1)
    uploaded_file = st.file_uploader("写真を撮る/選ぶ", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("2. 判定と計算結果")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        if st.button('判定してアドバイスを表示'):
            with st.spinner('計算中...'):
                candidate_labels = ["sweet potato", "chicken breast", "broccoli", "onion", "chocolate", "apple"]
                res = detector(image, candidate_labels=candidate_labels)
                top_label = res[0]['label']
                
                food_info = {
                    "sweet potato": {"name": "さつまいも", "kcal": 130, "icon": "🍠"},
                    "chicken breast": {"name": "ささみ", "kcal": 105, "icon": "🍗"},
                    "broccoli": {"name": "ブロッコリー", "kcal": 35, "icon": "🥦"},
                    "apple": {"name": "りんご", "kcal": 50, "icon": "🍎"},
                    "onion": {"name": "たまねぎ", "safe": False},
                    "chocolate": {"name": "チョコ", "safe": False}
                }

                if top_label in food_info:
                    info = food_info[top_label]
                    if info.get("safe", True):
                        # ユーザーが指定した量でのカロリー計算
                        input_kcal = (use_grams / 100) * info["kcal"]
                        # 上限値
                        daily_limit_kcal = dog_weight * 70 * 0.1
                        # 減らすべきドッグフード
                        reduce_food = input_kcal / 3.5
                        
                        st.success(f"判定：{info['icon']} {info['name']}")
                        
                        # 結果表示
                        st.metric(f"{info['name']} {use_grams}g のカロリー", f"{input_kcal:.1f} kcal")
                        
                        if input_kcal > daily_limit_kcal:
                            st.warning(f"⚠️ 少し多めです（目安は {daily_limit_kcal:.1f}kcal まで）")
                        else:
                            st.info("✅ 1日のトッピング制限内におさまっています。")
                        
                        st.write(f"👉 代わりにドッグフードを **約 {reduce_food:.1f}g** 減らしてあげましょう。")
                    else:
                        st.error(f"🚨 警告：{info['name']}は犬に与えてはいけません！")
    else:
        st.write("写真をアップロードしてください。")
