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
                
                # 上位2つの候補を取得
                top1 = res[0]
                top2 = res[1]
                
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

                # セッション状態で現在の選択を管理（初期値は1位）
                if 'selected_label' not in st.session_state:
                    st.session_state.selected_label = top1['label']

                # --- 候補選択ボタンの表示 ---
                st.write("🧐 AIの予想（違っていたら選んでください）:")
                c1, c2 = st.columns(2)
                with c1:
                    name1 = food_info[top1['label']]['name']
                    if st.button(f"1位: {name1} ({top1['score']:.1%})"):
                        st.session_state.selected_label = top1['label']
                with c2:
                    name2 = food_info[top2['label']]['name']
                    if st.button(f"2位: {name2} ({top2['score']:.1%})"):
                        st.session_state.selected_label = top2['label']

                # 選択されたラベルで計算を実行
                current_label = st.session_state.selected_label
                info = food_info[current_label]

                st.divider()
                
                   if info.get("safe", True):
                    input_kcal = (use_grams / 100) * info["kcal"]
                    daily_limit_kcal = dog_weight * 70 * 0.1
                    reduce_food = input_kcal / food_kcal_per_g
                    
                    st.success(f"判定中：{info['icon']} {info['name']}")
                    st.write(f"📊 **{info['name']} {use_grams}g** は **{input_kcal:.1f} kcal** です。")
                    
                    if input_kcal > daily_limit_kcal:
                        st.warning(f"⚠️ 少し多めです（目安は {daily_limit_kcal:.1f}kcal まで）")
                    else:
                        st.info("✅ 1日のトッピング制限内です。")
                    
                    st.subheader(f"🥣 調整量： **約 {reduce_food:.1f}g**")
                else:
                    st.error(f"🚨 警告：{info['name']}は危険です！")
