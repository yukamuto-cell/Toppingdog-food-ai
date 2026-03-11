import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. ページ全体の設定
st.set_page_config(page_title="DogFood AI", page_icon="🐶", layout="centered")

# デザインを整えるCSS
st.markdown("""
    <style>
    .stButton>button { border-radius: 20px; background-color: #ff4b4b; color: white; font-weight: bold; width: 100%; }
    .stRadio > label { font-weight: bold; padding-bottom: 10px; }
    .stNumberInput { margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🐶 愛犬トッピング・チェッカー")
st.write("AI判定 ＋ あなたの確認で、正確なカロリー調整を。")

# 2. AIモデルの準備（キャッシュを利用して高速化）
@st.cache_resource
def load_model():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")

detector = load_model()

# 食材データベースの定義
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

# --- 3. 入力エリア (左側) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 1. 設定")
    dog_weight = st.number_input("ワンちゃんの体重(kg)", min_value=0.1, value=5.0, step=0.1)
    
    food_kcal_per_100g = st.number_input("普段のフード (100gあたりkcal)", min_value=100, max_value=600, value=350)
    food_kcal_per_g = food_kcal_per_100g / 100
    
    use_grams = st.number_input("与えたい量(g)", min_value=1, value=10, step=1)
    uploaded_file = st.file_uploader("食材を撮影 / アップロード", type=["jpg", "jpeg", "png"])

# --- 4. 判定・結果エリア (右側) ---
with col2:
    st.subheader("🔍 2. 判定と結果")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        # 判定ボタン
        if st.button('AI判定を実行'):
            with st.spinner('AIが考えています...'):
                candidate_labels = list(food_info.keys())
                res = detector(image, candidate_labels=candidate_labels)
                # 上位2つの結果をセッションに保存
                st.session_state.preds = res[:2]

        # 判定結果が保存されている場合のみ表示
        if 'preds' in st.session_state:
            preds = st.session_state.preds
            
            st.write("🧐 正しい食材を選んでください:")
            
            # ラジオボタンの選択肢を作成
            options = [p['label'] for p in preds]
            option_display = {l: f"{food_info[l]['name']} ({next(p['score'] for p in preds if p['label'] == l):.1%})" for l in options}
            
            selected_label = st.radio(
                "AIの予測:",
                options,
                format_func=lambda x: option_display[x]
            )

            # --- 5. 計算結果の表示 ---
            st.divider()
            info = food_info[selected_label]
            
            if info.get("safe", True):
                # カロリー計算
                input_kcal = (use_grams / 100) * info["kcal"]
                daily_limit_kcal = dog_weight * 70 * 0.1
                reduce_food = input_kcal / food_kcal_per_g
                
                st.success(f"判定：{info['icon']} {info['name']}")
                
                # メトリック表示
                st.metric(f"{use_grams}g の摂取カロリー", f"{input_kcal:.1f} kcal")
                
                if input_kcal > daily_limit_kcal:
                    st.warning(f"⚠️ 制限量（{daily_limit_kcal:.1f}kcal）をオーバーしています")
                else:
                    st.info("✅ 1日の制限（10%）以内です")
                
                st.subheader(f"🥣 フード調整量： 約 {reduce_food:.1f} g")
                st.caption(f"（{food_kcal_per_100g}kcal/100g のフードの場合）")
            else:
                st.error(f"🚨 警告：{info['name']}は犬に危険です！")
    else:
        st.write("上に画像をアップロードすると判定が始まります。")
