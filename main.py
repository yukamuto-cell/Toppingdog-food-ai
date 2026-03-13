import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. ページ設定
st.set_page_config(page_title="DogFood AI Plus", page_icon="🐶", layout="centered")

st.markdown("""
    <style>
    .stButton>button { border-radius: 20px; background-color: #ff4b4b; color: white; font-weight: bold; width: 100%; }
    .stRadio > label { font-weight: bold; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🐶 愛犬ごはん調整チェッカー")

# 2. AIモデル準備
@st.cache_resource
def load_model():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")

detector = load_model()

# 食材データベース
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
    "egg": {"name": "たまご(ゆで)", "kcal": 150, "icon": "🥚"},
    "custom": {"name": "市販品(手入力)", "kcal": 0, "icon": "🍱"}
}

# --- 3. 入力エリア ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 1. 設定")
    dog_weight = st.number_input("ワンちゃんの体重(kg)", min_value=0.1, value=5.0, step=0.1)
    food_kcal_per_100g = st.number_input("メインフードのカロリー(100g/kcal)", min_value=100, value=350)
    food_kcal_per_g = food_kcal_per_100g / 100
    
    st.divider()
    
    calc_mode = st.radio(
        "ごはんの種類",
        ["トッピング(食材・一般食)", "総合栄養食(缶詰・パウチ等)"]
    )
    
    use_grams = st.number_input("今回の分量(g)", min_value=1, value=20, step=1)
    
    # 選択肢の初期値をモードに連動
    default_index = 0 if calc_mode == "トッピング(食材・一般食)" else 1
    input_method = st.radio("入力方法", ["画像で判定", "カロリーを手入力"], index=default_index)
    
    custom_kcal = 0
    uploaded_file = None
    
    if input_method == "画像で判定":
        uploaded_file = st.file_uploader("写真をアップロード", type=["jpg", "jpeg", "png"])
    else:
        custom_kcal = st.number_input("製品のカロリー(100g/kcal)", min_value=1, value=100)

# --- 4. 判定・結果エリア ---
with col2:
    st.subheader("🔍 2. 判定と結果")
    selected_label = None
    
    # 【新機能】総合栄養食モードで画像判定を選んでいる場合のアナウンス
    if calc_mode == "総合栄養食(缶詰・パウチ等)" and input_method == "画像で判定":
        st.warning("💡 総合栄養食（缶詰やパウチ）をお使いですね！")
        st.info("製品ごとにカロリーが異なるため、左のメニューで「カロリーを手入力」に切り替えて、パッケージ裏の数値を入力するのが正確でおすすめです。")
        
    if input_method == "画像で判定" and uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.last_file = uploaded_file.name
            with st.spinner('判定中...'):
                candidate_labels = [k for k in food_info.keys() if k != "custom"]
                res = detector(image, candidate_labels=candidate_labels)
                st.session_state.preds = res[:2]

        if 'preds' in st.session_state:
            preds = st.session_state.preds
            top_pred = preds[0]
            if top_pred['score'] >= 0.99:
                st.success(f"判定: {food_info[top_pred['label']]['name']}")
                selected_label = top_pred['label']
            else:
                options = [p['label'] for p in preds]
                option_display = {l: f"{food_info[l]['name']} ({next(p['score'] for p in preds if p['label'] == l):.1%})" for l in options}
                selected_label = st.radio("正しい食材を選択:", options, format_func=lambda x: option_display[x])
                
    elif input_method == "カロリーを手入力":
        selected_label = "custom"
        food_info["custom"]["kcal"] = custom_kcal
        st.success("✅ 数値入力モード")

    # --- 5. 計算結果の表示 ---
    if selected_label:
        st.divider()
        info = food_info[selected_label]
        
        if info.get("safe", True):
            input_kcal = (use_grams / 100) * info["kcal"]
            daily_limit_kcal = dog_weight * 70 * 0.1
            reduce_food = input_kcal / food_kcal_per_g
            
            st.metric(f"{info['name']} {use_grams}g", f"{input_kcal:.1f} kcal")
            
            if calc_mode == "トッピング(食材・一般食)":
                if input_kcal > daily_limit_kcal:
                    st.warning(f"⚠️ 制限目安：{daily_limit_kcal:.1f}kcal を超えています")
                else:
                    st.info("✅ 1日のトッピング制限内です")
            else:
                st.info("🥗 総合栄養食モード：栄養バランスは維持されます")
            
            st.subheader(f"🥣 メインを減らす量： 約 {reduce_food:.1f} g")
        else:
            st.error(f"🚨 警告：{info['name']}は危険です！")
    else:
        st.write("設定を完了してください。")
