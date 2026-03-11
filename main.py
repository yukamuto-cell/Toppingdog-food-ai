import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. ページ設定
st.set_page_config(page_title="DogFood AI", page_icon="🐶", layout="centered")

st.markdown("""
    <style>
    .stButton>button { border-radius: 20px; background-color: #ff4b4b; color: white; font-weight: bold; width: 100%; }
    .stRadio > label { font-weight: bold; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🐶 愛犬トッピング・チェッカー")

# 2. AIモデル準備
@st.cache_resource
def load_model():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")

detector = load_model()

food_info = {
    "sweet potato": {"name": "さつまいも", "kcal": 130, "icon": "🍠"},
    "chicken breast": {"name": "ささみ", "kcal": 105, "icon": "🍗"},
    "broccoli": {"name": "ブロッコリー", "kcal": 35, "icon": "🥦"},
    "apple": {"name": "りんご", "kcal": 50, "icon": "🍎"},
    "onion": {"name": "たまねぎ", "safe": False},
    "chocolate": {"name": "チョコ", "safe": False},
    "avocado": {"name": "アボカド", "safe": False},
    "yogurt": {"name": "ヨーグルト(無糖)", "kcal": 67, "icon": "🥛"},
    "salmon": {"name": "鮭(焼)", "kcal": 180, "icon": "🐟"},
    "pumpkin": {"name": "かぼちゃ", "kcal": 90, "icon": "🎃"},
    "egg": {"name": "たまご(ゆで)", "kcal": 150, "icon": "🥚"}
}

# --- 3. 入力エリア ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 1. 設定")
    dog_weight = st.number_input("ワンちゃんの体重(kg)", min_value=0.1, value=5.0, step=0.1)
    food_kcal_per_100g = st.number_input("普段のフード (100gあたりkcal)", min_value=100, max_value=600, value=350)
    food_kcal_per_g = food_kcal_per_100g / 100
    use_grams = st.number_input("与えたい量(g)", min_value=1, value=10, step=1)
    uploaded_file = st.file_uploader("食材をアップロード", type=["jpg", "jpeg", "png"])

# --- 4. 判定・結果エリア ---
with col2:
    st.subheader("🔍 2. 判定と結果")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        # 画像が変わったら判定をリセット
        if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.last_file = uploaded_file.name
            with st.spinner('AIが判定中...'):
                candidate_labels = list(food_info.keys())
                res = detector(image, candidate_labels=candidate_labels)
                st.session_state.preds = res[:2]

        if 'preds' in st.session_state:
            preds = st.session_state.preds
            top_pred = preds[0]
            
            # 【新機能】99%以上の自信がある場合は自動選択、そうでなければラジオボタン
            if top_pred['score'] >= 0.99:
                st.success(f"✨ 判定確定: {food_info[top_pred['label']]['name']} (自信: {top_pred['score']:.1%})")
                selected_label = top_pred['label']
            else:
                st.info("🧐 AIが少し迷っています。正しい方を選んでください:")
                options = [p['label'] for p in preds]
                option_display = {l: f"{food_info[l]['name']} ({next(p['score'] for p in preds if p['label'] == l):.1%})" for l in options}
                selected_label = st.radio("食材を選択:", options, format_func=lambda x: option_display[x])

            # --- 5. 計算結果の表示 ---
            st.divider()
            info = food_info[selected_label]
            
            if info.get("safe", True):
                input_kcal = (use_grams / 100) * info["kcal"]
                daily_limit_kcal = dog_weight * 70 * 0.1
                reduce_food = input_kcal / food_kcal_per_g
                
                st.metric(f"{info['icon']} {info['name']} {use_grams}g", f"{input_kcal:.1f} kcal")
                
                if input_kcal > daily_limit_kcal:
                    st.warning(f"⚠️ 制限量（{daily_limit_kcal:.1f}kcal）をオーバー")
                else:
                    st.info("✅ 制限内です")
                
                st.subheader(f"🥣 フード調整量： 約 {reduce_food:.1f} g")
            else:
                st.error(f"🚨 警告：{info['name']}は危険です！")
