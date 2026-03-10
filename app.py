import streamlit as st
import pandas as pd
import numpy as np
import joblib
import calendar
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import base64

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🛸 UFO Sighting Predictor 2026",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2. Background Image Processor
# ─────────────────────────────────────────────
def get_base64(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

bin_str = get_base64('645465453_1628827254986805_4766609008495813782_n.jpg')

# ─────────────────────────────────────────────
# 3. Custom CSS (Glassmorphism Style)
# ─────────────────────────────────────────────
if bin_str:
    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-attachment: fixed;
        background-size: cover;
    }}
    .main {{
        background: rgba(10, 10, 26, 0.6); 
    }}
    </style>
    """
else:
    bg_style = "<style>.main { background-color: #0a0a1a; }</style>"

st.markdown(bg_style, unsafe_allow_html=True)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { color: #00e5ff !important; text-shadow: 0 0 20px #00e5ff55; }
    h2, h3 { color: #7eb8f7 !important; }
    .metric-card {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid #00e5ff44;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 0 15px #00e5ff22;
        backdrop-filter: blur(8px);
    }
    .metric-number { font-size: 2.2rem; font-weight: 800; color: #00e5ff; }
    .metric-label  { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .shape-card {
        background: rgba(15, 52, 96, 0.7);
        border: 1px solid #7eb8f744;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 0.4rem 0;
        backdrop-filter: blur(5px);
    }
    .rank-badge {
        display:inline-block;
        background:#00e5ff22;
        border:1px solid #00e5ff;
        border-radius:50%;
        width:28px; height:28px;
        text-align:center; line-height:28px;
        font-weight:bold; color:#00e5ff;
        margin-right:8px;
    }
    .disclaimer {
        background: rgba(26, 26, 26, 0.85); border-left:4px solid #ff9800;
        padding:0.8rem 1rem; border-radius:0 8px 8px 0;
        font-size:0.82rem; color:#aaa; margin-top:2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 4. Model Loading
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_A  = joblib.load("model_monthly_count.pkl")
    model_B  = joblib.load("model_shape_classifier.pkl")
    model_C  = joblib.load("model_day_probability.pkl")
    le_shape = joblib.load("label_encoder_shape.pkl")
    le_day   = joblib.load("label_encoder_day.pkl")
    last_yr  = joblib.load("last_year_data.pkl")
    return model_A, model_B, model_C, le_shape, le_day, last_yr

try:
    model_A, model_B, model_C, le_shape, le_day, last_yr = load_models()
    models_ok = True
except Exception as e:
    models_ok = False
    st.error(f"❌ Error loading models: {e}")

# ─────────────────────────────────────────────
# 5. Constants & Helper Functions
# ─────────────────────────────────────────────
MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
MONTH_SHORT = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
SHAPE_EMOJI = {"light":"💡", "triangle":"🔺", "circle":"⭕", "fireball":"🔥", "disk":"💿", "sphere":"🔮", "other":"❓", "unknown":"❔"}
SHAPE_DESC = {"light": "แสงสว่างลอยอยู่ในอากาศ", "triangle": "ทรงสามเหลี่ยม", "circle": "ทรงกลม/วงกลม", "fireball": "ลูกไฟ", "disk": "จานบิน", "sphere": "ทรงกลมสมบูรณ์", "other": "รูปร่างอื่นๆ", "unknown": "ไม่ทราบรูปร่าง"}
SEASON_MAP = {12:4,1:4,2:4,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}

def predict_monthly_count(month, year=2026):
    season, is_summer, is_holiday = SEASON_MAP[month], (1 if month in [6,7,8] else 0), (1 if month in [7,12] else 0)
    count_lag1 = last_yr.get(month-1 if month>1 else 12, last_yr.mean())
    count_lag12 = last_yr.get(month, last_yr.mean())
    row = pd.DataFrame([{"month": month, "season": season, "is_summer": is_summer, "is_holiday_month": is_holiday, "year_since_1990": year-1990, "count_lag1": count_lag1, "count_lag12": count_lag12, "rolling_mean_3": last_yr.mean()}])
    return max(1, int(round(model_A.predict(row)[0])))

def predict_shapes(month):
    rf_model, scaler_B = model_B.named_steps["model"], model_B.named_steps["scaler"]
    proba_total = np.zeros(len(le_shape.classes_))
    for h, wd in [(h, wd) for h in [20, 21, 22, 23] for wd in range(7)]:
        row = pd.DataFrame([{"month": month, "day": 15, "hour": h, "weekday": wd, "season": SEASON_MAP[month], "is_summer": 1 if month in [6,7,8] else 0, "is_holiday_month": 1 if month in [7,12] else 0, "is_night": 1}])
        proba_total += rf_model.predict_proba(scaler_B.transform(row))[0]
    proba_avg = proba_total / 28
    top_idx = np.argsort(proba_avg)[::-1][:3]
    return [{"shape": le_shape.classes_[i], "prob": proba_avg[i]} for i in top_idx], proba_avg

def predict_days(month, year=2026):
    num_days = calendar.monthrange(year, month)[1]
    res = []
    for d in range(1, num_days + 1):
        row = pd.DataFrame([{"month": month, "day": d, "is_summer": 1 if month in [6,7,8] else 0, "is_holiday_month": 1 if month in [7,12] else 0, "season": SEASON_MAP[month]}])
        res.append({"day": d, "level": le_day.inverse_transform([model_C.predict(row)[0]])[0]})
    return pd.DataFrame(res)

# ─────────────────────────────────────────────
# 6. Main UI Logic
# ─────────────────────────────────────────────
if models_ok:
    with st.sidebar:
        st.markdown("## 🔭 Control Center")
        selected_month = st.selectbox("เลือกเดือนพยากรณ์", options=list(range(1, 13)), format_func=lambda x: f"{x:02d} — {MONTH_NAMES[x-1]}", index=6)
        show_all_months = st.checkbox("แสดงรายปี 2026", value=True)
        show_feature_imp = st.checkbox("Feature Importance", value=True)

    st.markdown("<h1 style='text-align:center;'>🛸 UFO Sighting Predictor 2026</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:#7eb8f7;'>พยากรณ์ข้อมูลทางสถิติสำหรับเดือน {MONTH_NAMES[selected_month-1]}</p>", unsafe_allow_html=True)

    # Calculation
    pred_count = predict_monthly_count(selected_month)
    shape_results, _ = predict_shapes(selected_month)
    df_days = predict_days(selected_month)
    all_counts = [predict_monthly_count(m) for m in range(1, 13)]
    high_days = df_days[df_days["level"] == "high"]["day"].tolist()

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-number'>{pred_count}</div><div class='metric-label'>คาดการณ์การพบเห็น</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-number'>{SHAPE_EMOJI.get(shape_results[0]['shape'],'🛸')}</div><div class='metric-label'>รูปทรงหลัก: {shape_results[0]['shape'].upper()}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-number'>{len(high_days)}</div><div class='metric-label'>วันที่มีโอกาสสูง (🔴)</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-number'>{MONTH_SHORT[np.argmax(all_counts)]}</div><div class='metric-label'>เดือนที่พบมากที่สุด</div></div>", unsafe_allow_html=True)

    # Columns for Shape & Importance
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"### 🔮 รูปทรงที่น่าจะปรากฏ")
        for i, s in enumerate(shape_results):
            pct = s['prob'] * 100
            st.markdown(f"<div class='shape-card'><span class='rank-badge'>{i+1}</span><b>{SHAPE_EMOJI.get(s['shape'],'🛸')} {s['shape'].upper()}</b> ({pct:.1f}%)<br><small style='color:#aaa'>{SHAPE_DESC.get(s['shape'],'')}</small></div>", unsafe_allow_html=True)
    
    with col_right:
        if show_feature_imp:
            st.markdown("### 📌 Feature Importance")
            rf_B = model_B.named_steps["model"]
            feats = ["month","day","hour","weekday","season","is_summer","is_holiday","is_night"]
            fig_fi, ax = plt.subplots(figsize=(6, 4.2))
            fig_fi.patch.set_alpha(0)
            ax.set_facecolor("none")
            ax.barh(feats, rf_B.feature_importances_, color="#00e5ff")
            ax.tick_params(colors="white")
            st.pyplot(fig_fi)

    # Calendar Section
    st.markdown(f"### 📅 ตารางความน่าจะเป็นรายวัน: {MONTH_NAMES[selected_month-1]}")
    first_wd, num_days = calendar.monthrange(2026, selected_month)
    fig_cal, ax = plt.subplots(figsize=(10, 4))
    fig_cal.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(0, 7); ax.set_ylim(0, 6); ax.axis("off")
    
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for i, d in enumerate(days): ax.text(i+0.5, 5.5, d, color="white", ha='center', fontweight='bold')
    
    day_map = dict(zip(df_days['day'], df_days['level']))
    row, col = 4, first_wd
    for d in range(1, num_days+1):
        lvl = day_map.get(d, 'low')
        color = "#FF4444" if lvl=='high' else ("#FFA500" if lvl=='medium' else "#1a8a1a")
        rect = plt.Rectangle((col, row), 0.9, 0.8, color=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(col+0.45, row+0.4, str(d), color="white", ha='center', va='center', fontweight='bold')
        col += 1
        if col > 6: col = 0; row -= 1
    st.pyplot(fig_cal)

    # Yearly Chart
    if show_all_months:
        st.markdown("### 📈 พยากรณ์ภาพรวมปี 2026")
        fig_yr, ax = plt.subplots(figsize=(12, 3))
        fig_yr.patch.set_alpha(0)
        ax.set_facecolor("none")
        colors = ["#FF6B6B" if m in [6,7,8] else "#2a4a7a" for m in range(1,13)]
        colors[selected_month-1] = "#00e5ff"
        ax.bar(MONTH_SHORT, all_counts, color=colors)
        ax.tick_params(colors="white")
        st.pyplot(fig_yr)

st.markdown("<div class='disclaimer'>⚠️ <b>Disclaimer:</b> ข้อมูลนี้สร้างจากโมเดลสถิติ AI (NUFORC Data) ไม่ใช่หลักฐานยืนยันทางวิทยาศาสตร์เกี่ยวกับการมีอยู่ของสิ่งมีชีวิตนอกโลก</div>", unsafe_allow_html=True)
