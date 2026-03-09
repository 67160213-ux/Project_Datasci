import streamlit as st
import pandas as pd
import numpy as np
import joblib
import calendar
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🛸 UFO Sighting Predictor 2026",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a1a; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #00e5ff !important; text-shadow: 0 0 20px #00e5ff55; }
    h2, h3 { color: #7eb8f7 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #00e5ff44;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 0 15px #00e5ff22;
    }
    .metric-number { font-size: 2.2rem; font-weight: 800; color: #00e5ff; }
    .metric-label  { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .shape-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        border: 1px solid #7eb8f744;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 0.4rem 0;
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
        background:#1a1a1a; border-left:4px solid #ff9800;
        padding:0.8rem 1rem; border-radius:0 8px 8px 0;
        font-size:0.82rem; color:#aaa; margin-top:1rem;
    }
    .stSlider > div > div { color: #00e5ff; }
    div[data-testid="stMetric"] { background:#1a1a2e; border-radius:10px; padding:0.6rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load models
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
    st.error(f"❌ ไม่พบไฟล์โมเดล: {e}\nกรุณาวางไฟล์ .pkl ทั้งหมดไว้ในโฟลเดอร์เดียวกับ app.py")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
MONTH_SHORT = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
SHAPE_EMOJI = {
    "light":"💡", "triangle":"🔺", "circle":"⭕",
    "fireball":"🔥", "disk":"💿", "sphere":"🔮",
    "other":"❓", "unknown":"❔"
}
SHAPE_DESC = {
    "light":    "แสงสว่างลอยอยู่ในอากาศ",
    "triangle": "ทรงสามเหลี่ยม",
    "circle":   "ทรงกลม/วงกลม",
    "fireball": "ลูกไฟ",
    "disk":     "จานบิน",
    "sphere":   "ทรงกลมสมบูรณ์",
    "other":    "รูปร่างอื่นๆ",
    "unknown":  "ไม่ทราบรูปร่าง",
}
SEASON_MAP = {12:4,1:4,2:4,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def predict_monthly_count(month: int, year: int = 2026):
    season      = SEASON_MAP[month]
    is_summer   = 1 if month in [6,7,8] else 0
    is_holiday  = 1 if month in [7,12] else 0
    lag1_month  = month - 1 if month > 1 else 12
    count_lag1  = last_yr.get(lag1_month, last_yr.mean())
    count_lag12 = last_yr.get(month,      last_yr.mean())
    rolling_mean_3 = last_yr.mean()
    row = pd.DataFrame([{
        "month": month, "season": season,
        "is_summer": is_summer, "is_holiday_month": is_holiday,
        "year_since_1990": year - 1990,
        "count_lag1": count_lag1,
        "count_lag12": count_lag12,
        "rolling_mean_3": rolling_mean_3,
    }])
    pred = model_A.predict(row)[0]
    return max(1, int(round(pred)))


def predict_shapes(month: int):
    season     = SEASON_MAP[month]
    is_summer  = 1 if month in [6,7,8] else 0
    is_holiday = 1 if month in [7,12] else 0
    rf_model = model_B.named_steps["model"]
    scaler_B = model_B.named_steps["scaler"]
    proba_total = np.zeros(len(le_shape.classes_))
    for h in [20, 21, 22, 23]:
        for wd in range(7):
            row = pd.DataFrame([{
                "month": month, "day": 15, "hour": h, "weekday": wd,
                "season": season, "is_summer": is_summer,
                "is_holiday_month": is_holiday, "is_night": 1,
            }])
            row_s = scaler_B.transform(row)
            proba_total += rf_model.predict_proba(row_s)[0]
    proba_avg = proba_total / (4 * 7)
    top3_idx  = np.argsort(proba_avg)[::-1][:3]
    results = []
    for idx in top3_idx:
        results.append({
            "shape":    le_shape.classes_[idx],
            "prob":     proba_avg[idx],
        })
    return results, proba_avg


def predict_days(month: int, year: int = 2026):
    season     = SEASON_MAP[month]
    is_summer  = 1 if month in [6,7,8] else 0
    is_holiday = 1 if month in [7,12] else 0
    num_days   = calendar.monthrange(year, month)[1]
    results = []
    for d in range(1, num_days + 1):
        row = pd.DataFrame([{
            "month": month, "day": d,
            "is_summer": is_summer, "is_holiday_month": is_holiday,
            "season": season,
        }])
        enc = model_C.predict(row)[0]
        level = le_day.inverse_transform([enc])[0]
        results.append({"day": d, "level": level})
    return pd.DataFrame(results)


def predict_all_months(year: int = 2026):
    return [predict_monthly_count(m, year) for m in range(1, 13)]


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛸 UFO Predictor 2026")
    st.markdown("---")

    st.markdown("### 🔭 เลือกเดือนที่ต้องการทำนาย")
    selected_month = st.selectbox(
        "เดือน",
        options=list(range(1, 13)),
        format_func=lambda x: f"{x:02d} — {MONTH_NAMES[x-1]}",
        index=6,
    )

    st.markdown("### ⚙️ ตัวเลือกเพิ่มเติม")
    show_all_months = st.checkbox("แสดงการพยากรณ์ทั้งปี 2026", value=True)
    show_feature_imp = st.checkbox("แสดง Feature Importance", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#666;'>
    <b>Dataset:</b> NUFORC UFO Sightings<br>
    <b>Period:</b> 1990–2013 (US)<br>
    <b>Records:</b> ~62,000 entries<br><br>
    <b>Models:</b><br>
    • 🟢 Random Forest Regressor<br>
    • 🔵 Random Forest Classifier<br>
    • 🟠 Gradient Boosting Classifier
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; font-size:2.4rem; margin-bottom:0.2rem;'>
    🛸 UFO Sighting Predictor
</h1>
<p style='text-align:center; color:#7eb8f7; font-size:1.05rem; margin-bottom:1.5rem;'>
    ทำนายการพบเห็น UFO ในสหรัฐอเมริกา ปี 2026 — ด้วย Machine Learning
</p>
""", unsafe_allow_html=True)

if not models_ok:
    st.stop()

# ─────────────────────────────────────────────
# Run predictions
# ─────────────────────────────────────────────
pred_count   = predict_monthly_count(selected_month)
shape_results, shape_proba = predict_shapes(selected_month)
df_days      = predict_days(selected_month)
all_counts   = predict_all_months()

month_name   = MONTH_NAMES[selected_month - 1]
lo, hi       = int(pred_count * 0.85), int(pred_count * 1.15)

high_days   = df_days[df_days["level"] == "high"]["day"].tolist()
medium_days = df_days[df_days["level"] == "medium"]["day"].tolist()
low_days    = df_days[df_days["level"] == "low"]["day"].tolist()

# ─────────────────────────────────────────────
# Top KPI cards
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-number'>{pred_count:,}</div>
        <div class='metric-label'>🛸 การพบ UFO ใน {month_name}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    top_shape = shape_results[0]["shape"]
    top_emoji = SHAPE_EMOJI.get(top_shape, "🛸")
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-number'>{top_emoji} {top_shape.upper()}</div>
        <div class='metric-label'>🔮 รูปร่างที่น่าจะพบมากที่สุด</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-number'>{len(high_days)}</div>
        <div class='metric-label'>🔴 วันที่มีโอกาสพบสูง</div>
    </div>""", unsafe_allow_html=True)

with c4:
    peak_month_idx = int(np.argmax(all_counts)) + 1
    peak_name      = MONTH_SHORT[peak_month_idx - 1]
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-number'>{peak_name}</div>
        <div class='metric-label'>📈 เดือนที่คาดว่าพบมากที่สุดในปี 2026</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Section 1: Monthly prediction detail
# ─────────────────────────────────────────────
st.markdown(f"## 📊 การพยากรณ์เดือน {month_name} 2026")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # Count detail
    st.markdown(f"""
    <div class='metric-card' style='text-align:left;'>
        <div style='color:#aaa; font-size:0.85rem; margin-bottom:0.6rem;'>📌 จำนวนการพบที่ทำนาย</div>
        <div style='font-size:2rem; font-weight:800; color:#00e5ff;'>{pred_count:,} ครั้ง</div>
        <div style='color:#7eb8f7; font-size:0.9rem; margin-top:0.3rem;'>
            ช่วงความไม่แน่นอน: {lo:,} – {hi:,} ครั้ง
        </div>
        <hr style='border-color:#333; margin:0.8rem 0;'>
        <div style='color:#aaa; font-size:0.82rem;'>
            🌞 เดือนนี้{'เป็น <b style="color:#FF6B6B">ช่วง Summer</b> — โอกาสพบสูงกว่าค่าเฉลี่ย' if selected_month in [6,7,8] else 'ไม่ใช่ช่วง Summer'}<br>
            📅 {'เป็น Holiday Month (Jul/Dec)' if selected_month in [7,12] else 'ไม่ใช่ Holiday Month'}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Shape predictions
    st.markdown("### 🔮 รูปร่าง UFO ที่น่าจะพบ")
    for i, s in enumerate(shape_results):
        emoji  = SHAPE_EMOJI.get(s["shape"], "🛸")
        desc   = SHAPE_DESC.get(s["shape"], "")
        pct    = s["prob"] * 100
        bar_w  = int(pct * 3)
        st.markdown(f"""
        <div class='shape-card'>
            <span class='rank-badge'>{i+1}</span>
            <b style='color:#fff; font-size:1.05rem;'>{emoji} {s["shape"].upper()}</b>
            <span style='color:#aaa; font-size:0.82rem; margin-left:8px;'>— {desc}</span><br>
            <div style='margin-top:0.4rem; background:#222; border-radius:10px; height:8px; overflow:hidden;'>
                <div style='width:{min(bar_w,100)}%; background:linear-gradient(90deg,#00e5ff,#7eb8f7); height:100%; border-radius:10px;'></div>
            </div>
            <div style='color:#7eb8f7; font-size:0.85rem; margin-top:0.25rem;'>{pct:.1f}% probability</div>
        </div>""", unsafe_allow_html=True)

with col_right:
    # Feature importance chart
    if show_feature_imp:
        st.markdown("### 📌 Feature Importance (Shape Model)")
        rf_B       = model_B.named_steps["model"]
        feat_names = ["month","day","hour","weekday","season","is_summer","is_holiday","is_night"]
        importances = rf_B.feature_importances_
        idx_sort    = np.argsort(importances)

        fig_fi, ax = plt.subplots(figsize=(6, 4))
        fig_fi.patch.set_facecolor("#0a0a1a")
        ax.set_facecolor("#0a0a1a")
        colors = ["#00e5ff" if i == idx_sort[-1] else "#2a4a7a" for i in range(len(feat_names))]
        bars = ax.barh([feat_names[i] for i in idx_sort], importances[idx_sort],
                       color=[colors[i] for i in idx_sort], edgecolor="none")
        ax.set_xlabel("Importance Score", color="#aaa", fontsize=9)
        ax.tick_params(colors="#ccc", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.xaxis.label.set_color("#aaa")
        st.pyplot(fig_fi, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────
# Section 2: Calendar heatmap
# ─────────────────────────────────────────────
st.markdown(f"## 📅 ปฏิทินโอกาสพบ UFO — {month_name} 2026")
st.markdown("""
<p style='color:#aaa; font-size:0.85rem;'>
🔴 HIGH = โอกาสพบสูง &nbsp;&nbsp;|&nbsp;&nbsp;
🟡 MEDIUM = โอกาสพบปานกลาง &nbsp;&nbsp;|&nbsp;&nbsp;
🟢 LOW = โอกาสพบต่ำ
</p>""", unsafe_allow_html=True)

first_wd  = calendar.monthrange(2026, selected_month)[0]  # 0=Mon
num_days  = calendar.monthrange(2026, selected_month)[1]
level_colors = {"high": "#FF4444", "medium": "#FFA500", "low": "#1a8a1a"}
level_alpha  = {"high": 0.85, "medium": 0.75, "low": 0.60}

fig_cal, ax = plt.subplots(figsize=(12, 4))
fig_cal.patch.set_facecolor("#0a0a1a")
ax.set_facecolor("#0a0a1a")
ax.set_xlim(-0.1, 7.1)
ax.set_ylim(-0.6, 7.1)
ax.axis("off")

weekday_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
for i, wd in enumerate(weekday_labels):
    color = "#FF6B6B" if wd in ["Sat","Sun"] else "#7eb8f7"
    ax.text(i + 0.45, 6.65, wd, ha="center", va="center",
            fontsize=11, fontweight="bold", color=color)

day_level_map = dict(zip(df_days["day"], df_days["level"]))
col = first_wd
row = 5

for day in range(1, num_days + 1):
    level  = day_level_map.get(day, "low")
    color  = level_colors[level]
    alpha  = level_alpha[level]
    rect   = plt.Rectangle((col + 0.04, row - 0.88), 0.88, 0.76,
                            facecolor=color, alpha=alpha,
                            edgecolor="#0a0a1a", linewidth=2.5, zorder=2)
    ax.add_patch(rect)
    ax.text(col + 0.48, row - 0.47, str(day),
            ha="center", va="center", fontsize=12,
            fontweight="bold", color="white", zorder=3)
    if level == "high":
        ax.text(col + 0.48, row - 0.78, "🛸",
                ha="center", va="center", fontsize=7, zorder=3)
    col += 1
    if col == 7:
        col = 0
        row -= 1

# Legend
legend_items = [
    mpatches.Patch(color="#FF4444", alpha=0.85, label=f"HIGH — {len(high_days)} วัน"),
    mpatches.Patch(color="#FFA500", alpha=0.75, label=f"MEDIUM — {len(medium_days)} วัน"),
    mpatches.Patch(color="#1a8a1a", alpha=0.60, label=f"LOW — {len(low_days)} วัน"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=10,
          facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

st.pyplot(fig_cal, use_container_width=True)
plt.close()

# ─────────────────────────────────────────────
# Section 3: All-year forecast
# ─────────────────────────────────────────────
if show_all_months:
    st.markdown("## 📈 การพยากรณ์ทั้งปี 2026")

    fig_yr, ax = plt.subplots(figsize=(13, 4))
    fig_yr.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    x = np.arange(1, 13)
    y = np.array(all_counts, dtype=float)
    bar_colors = ["#FF6B6B" if m in [6,7,8] else
                  ("#7eb8f7" if m == selected_month else "#2a4a7a")
                  for m in range(1, 13)]

    bars = ax.bar(x, y, color=bar_colors, edgecolor="#0a0a1a", linewidth=1.5, zorder=3)

    # Highlight selected
    bars[selected_month - 1].set_edgecolor("#00e5ff")
    bars[selected_month - 1].set_linewidth(3)

    # Labels on top
    for bar, val in zip(bars, all_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:,}", ha="center", va="bottom",
                fontsize=8.5, color="#ccc", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_SHORT, color="#ccc", fontsize=10)
    ax.set_ylabel("Predicted Sightings", color="#aaa", fontsize=9)
    ax.tick_params(axis="y", colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_facecolor("#0a0a1a")
    ax.grid(axis="y", color="#222", linewidth=0.8, zorder=0)

    legend_handles = [
        mpatches.Patch(color="#FF6B6B", label="Summer (Jun–Aug)"),
        mpatches.Patch(color="#7eb8f7", label=f"Selected: {month_name}"),
        mpatches.Patch(color="#2a4a7a", label="Other months"),
    ]
    ax.legend(handles=legend_handles, facecolor="#1a1a2e",
              edgecolor="#333", labelcolor="white", fontsize=9)

    st.pyplot(fig_yr, use_container_width=True)
    plt.close()

    # Table
    df_table = pd.DataFrame({
        "เดือน":          MONTH_NAMES,
        "ทำนาย (ครั้ง)":  all_counts,
        "ระดับ":          ["🔴 สูงมาก" if c == max(all_counts)
                          else "🟠 สูง" if c >= np.percentile(all_counts, 75)
                          else "🟡 กลาง" if c >= np.percentile(all_counts, 25)
                          else "🟢 ต่ำ"
                          for c in all_counts],
    })
    df_table.index = range(1, 13)
    st.dataframe(df_table, use_container_width=True, height=250)

# ─────────────────────────────────────────────
# Section 4: High-day detail
# ─────────────────────────────────────────────
st.markdown(f"## 🔴 วันที่ควรจับตามอง — {month_name} 2026")

cols = st.columns(3)
day_groups = [
    ("🔴 HIGH — โอกาสพบสูง",   high_days,   "#FF4444"),
    ("🟡 MEDIUM — ปานกลาง",    medium_days, "#FFA500"),
    ("🟢 LOW — โอกาสพบต่ำ",    low_days,    "#1a8a1a"),
]
for col, (title, days, color) in zip(cols, day_groups):
    with col:
        day_str = ", ".join([f"**{d}**" for d in days]) if days else "—"
        st.markdown(f"""
        <div class='metric-card' style='text-align:left;'>
            <div style='color:{color}; font-weight:bold; margin-bottom:0.5rem;'>{title}</div>
            <div style='color:#ddd; font-size:0.9rem; line-height:1.8;'>
                {", ".join([str(d) for d in days]) if days else "—"}
            </div>
            <div style='color:#666; font-size:0.78rem; margin-top:0.5rem;'>{len(days)} วัน จากทั้งหมด {num_days} วัน</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Disclaimer
# ─────────────────────────────────────────────
st.markdown("""
<div class='disclaimer'>
⚠️ <b>Disclaimer:</b> การทำนายนี้สร้างจากโมเดล Machine Learning ที่เรียนรู้จากข้อมูลประวัติการรายงาน UFO 
(NUFORC, 1990–2013) เพื่อวัตถุประสงค์ทางการศึกษาเท่านั้น 
ผลลัพธ์ไม่ใช่หลักฐานทางวิทยาศาสตร์ และไม่ยืนยันการมีอยู่จริงของยานอวกาศต่างดาว
</div>
""", unsafe_allow_html=True)
