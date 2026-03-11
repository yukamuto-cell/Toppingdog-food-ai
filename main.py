import streamlit as st
from transformers import pipeline
from PIL import Image

# ページ設定
st.set_page_config(page_title="DogFood AI", page_icon="🐶")

# デザイン調整
st.markdown("""
    <style>
    .stButton>button { border-radius: 20px; background-color: #ff4b4b; color: white; font-weight: bold; }
    .stNumberInput { margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🐶 愛犬トッピング・チェッカー")
st.write("普段のフードに合わせて、正確な調整量を計算します。")

# AIモデルの準備
@st.cache_resource
def load_model():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")

detector = load_model()

# --- 入力エリア ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 1. 設定")
    dog_weight = st.number_input("ワンちゃんの体重(kg)", min_value=0.1, value=5.0, step=0.1)
    
    # フードのカロリー入力
    food_kcal_per_100g = st.number_input("フードのカロリー (100gあたり/kcal)", min_value=100, max_value=600, value=350)
    # 1gあたりのカロリーを算出
    food_kcal_per_g = food_kcal_per_100g / 100
    
    use_grams = st.number_input("与えたいトッピングの分量(g)", min_value=1, value=10, step=1)
    uploaded_file = st.file_uploader("食材の写真をアップロード", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("🔍 2. 判定と結果")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        # ここから下のインデント（スペース）を修正しました
        if st.button('計算を実行する'):
            with st.spinner('判定中...'):
                candidate_labels = [
                    "sweet potato", "chicken breast", "broccoli", "onion", 
                    "chocolate", "apple", "avocado", "yogurt", "salmon", 
                    "pumpkin", "egg"
                ]
                res = detector(image, candidate_labels=candidate_labels)
                top_label = res[0]['label']
                
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

                if top_label in food_info:
                    info = food_info[top_label]
                    if info.get("safe", True):
                        # トッピングの合計kcal
                        input_kcal = (use_grams / 100) * info["kcal"]
                        # 目標上限値
                        daily_limit_kcal = dog_weight * 70 * 0.1
                        
                        # 【重要】設定されたフードのカロリーで計算
                        reduce_food = input_kcal / food_kcal_per_g
                        
                        st.success(f"判定：{info['icon']} {info['name']}")
                        
                        # 結果表示
                        st.write(f"📊 **{info['name']} {use_grams}g** は **{input_kcal:.1f} kcal** です。")
                        
                        # 判定メッセージ
                        if input_kcal > daily_limit_kcal:
                            st.warning(f"⚠️ 少し多めです（目安は {daily_limit_kcal:.1f}kcal まで）")
                        else:
                            st.info("✅ 1日のトッピング制限内です。")
                        
                        # 最終的なアドバイス
                        st.divider()
                        st.subheader(f"🥣 調整量： **約 {reduce_food:.1f}g**")
                        st.write(f"普段のフード（{food_kcal_per_100g}kcal/100g）をこの分だけ減らしてください。")
                    else:
                        st.error(f"🚨 警告：{info['name']}は危険です！")
    else:
        st.write("写真をアップロードしてください。")
