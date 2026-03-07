import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joblib
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛸 UFO Sightings Forecaster",
    page_icon="🛸",
    layout="wide",
)

# ─── Load Model & Metadata ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("ufo_model.pkl")
    features = joblib.load("ufo_features.pkl")
    meta     = joblib.load("ufo_meta.pkl")
    # โหลด RF แยกสำหรับ Feature Importance (มี feature_importances_ เสมอ)
    rf_model = joblib.load("ufo_rf_model.pkl") if os.path.exists("ufo_rf_model.pkl") else model
    return model, features, meta, rf_model

@st.cache_data
def load_data():
    df = pd.read_csv("scrubbed.csv", low_memory=False)
    df['shape']    = df['shape'].fillna('unknown')
    df['country']  = df['country'].fillna('unknown')
    df['latitude']           = pd.to_numeric(df['latitude'].astype(str).str.replace('[^0-9.-]','',regex=True), errors='coerce')
    df['duration (seconds)'] = pd.to_numeric(df['duration (seconds)'].astype(str).str.replace('[^0-9.-]','',regex=True), errors='coerce')
    df['datetime'] = pd.to_datetime(
        df['datetime'].astype(str).str.replace('24:00','00:00'),
        format='%m/%d/%Y %H:%M', errors='coerce'
    )
    df = df.dropna(subset=['datetime'])
    df['year']  = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    return df

try:
    model, features, meta, rf_model = load_model()
    df = load_data()
    model_loaded = True
except Exception as e:
    model_loaded = False
    rf_model = None
    st.warning(f"⚠️ ยังไม่พบ model file — กรุณา run notebook ก่อน ({e})")
    df = load_data()

# ─── Helper: build feature vector ──────────────────────────────────────────────
def build_features(year: int, month: int, history_series: pd.Series) -> pd.DataFrame:
    """สร้าง feature vector สำหรับ 1 เดือน"""
    t_val = (year - 1990) * 12 + (month - 1)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    lag_1  = history_series.iloc[-1]  if len(history_series) >= 1  else 0
    lag_2  = history_series.iloc[-2]  if len(history_series) >= 2  else 0
    lag_12 = history_series.iloc[-12] if len(history_series) >= 12 else 0

    rolling_3  = history_series.iloc[-3:].mean()  if len(history_series) >= 3  else lag_1
    rolling_12 = history_series.iloc[-12:].mean() if len(history_series) >= 12 else lag_1

    return pd.DataFrame([[t_val, month_sin, month_cos, lag_1, lag_2, lag_12, rolling_3, rolling_12]],
                        columns=features)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Simple_disc_shaped_UFO.svg/300px-Simple_disc_shaped_UFO.svg.png", width=100)
    st.title("🛸 UFO Forecaster")
    st.markdown("---")
    st.markdown("**เกี่ยวกับแอปนี้**")
    st.info(
        "แอปนี้ใช้ Machine Learning ทำนายจำนวนการพบ UFO รายเดือน "
        "โดยเรียนรู้จากข้อมูลย้อนหลัง 1990–2014 จาก NUFORC"
    )
    page = st.radio("เลือกหน้า", ["🔮 ทำนาย", "📊 Feature Importance", "📈 ข้อมูล EDA"])
    st.markdown("---")
    if model_loaded:
        st.success(f"✅ Model: `{meta['best_model']}`")
        r2 = meta['metrics'][meta['best_model']]['R2']
        mae = meta['metrics'][meta['best_model']]['MAE']
        st.metric("R² Score", f"{r2:.3f}")
        st.metric("MAE", f"{mae:.1f} sightings/month")
    st.markdown("---")
    st.caption("⚠️ Disclaimer: แอปนี้สร้างเพื่อการศึกษาเท่านั้น ผลการทำนายอาจไม่สะท้อนความเป็นจริง")

# ─── Page 1: Prediction ────────────────────────────────────────────────────────
if page == "🔮 ทำนาย":
    st.title("🔮 ทำนายจำนวนการพบ UFO")
    st.markdown("ป้อนข้อมูลด้านล่างเพื่อทำนายจำนวนการพบ UFO ในเดือนที่ต้องการ")

    col1, col2 = st.columns(2)
    with col1:
        pred_year = st.slider("ปี (Year)", min_value=2015, max_value=2030, value=2020,
                              help="เลือกปีที่ต้องการทำนาย (โมเดล train จากข้อมูลถึงปี 2014)")
    with col2:
        pred_month = st.slider("เดือน (Month)", min_value=1, max_value=12, value=7,
                               help="1=มกราคม … 12=ธันวาคม")

    month_names = ['มกราคม','กุมภาพันธ์','มีนาคม','เมษายน','พฤษภาคม','มิถุนายน',
                   'กรกฎาคม','สิงหาคม','กันยายน','ตุลาคม','พฤศจิกายน','ธันวาคม']

    if st.button("🚀 ทำนาย", type="primary", use_container_width=True):
        if not model_loaded:
            st.error("กรุณา run notebook และโหลด model ก่อน")
        else:
            # สร้าง history จากข้อมูลจริง
            monthly_ts = df.groupby(['year','month']).size().reset_index(name='count')
            monthly_ts = monthly_ts[(monthly_ts['year'] >= 1990)].sort_values(['year','month'])
            history    = monthly_ts['count'].values

            X_input = build_features(pred_year, pred_month, pd.Series(history))
            pred    = model.predict(X_input)[0]
            pred    = max(0, int(round(pred)))

            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("📅 เดือนที่ทำนาย", f"{month_names[pred_month-1]} {pred_year}")
            col_b.metric("🛸 จำนวนที่ทำนาย", f"{pred:,} ครั้ง")
            col_c.metric("📊 R² ของโมเดล", f"{meta['metrics'][meta['best_model']]['R2']:.3f}")

            # เปรียบเทียบกับค่าเฉลี่ยในเดือนเดียวกัน
            same_month_avg = df[df['month'] == pred_month].groupby('year').size().mean()
            diff = pred - same_month_avg
            sign = "+" if diff >= 0 else ""
            st.info(f"เปรียบเทียบกับค่าเฉลี่ยเดือน{month_names[pred_month-1]} ในอดีต: "
                    f"**{same_month_avg:.0f}** ครั้ง  →  ต่างกัน **{sign}{diff:.0f}** ครั้ง")

            # แสดง trend ย้อนหลัง
            st.subheader("📈 ประวัติและค่าที่ทำนาย")
            hist_monthly = df.groupby(['year','month']).size().reset_index(name='count')
            hist_monthly = hist_monthly[hist_monthly['year'] >= 2005].sort_values(['year','month'])
            hist_monthly['date'] = pd.to_datetime(hist_monthly[['year','month']].assign(day=1))

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(hist_monthly['date'], hist_monthly['count'], color='steelblue', linewidth=1.5, label='ข้อมูลจริง')
            future_date = pd.Timestamp(year=pred_year, month=pred_month, day=1)
            ax.scatter([future_date], [pred], color='tomato', s=150, zorder=5, label=f'ทำนาย ({pred:,} ครั้ง)')
            ax.axvline(pd.Timestamp('2015-01-01'), color='gray', linestyle='--', alpha=0.5, label='สิ้นสุดข้อมูล')
            ax.set_xlabel('เดือน')
            ax.set_ylabel('จำนวนการพบ UFO')
            ax.set_title('ประวัติการพบ UFO และค่าที่ทำนาย')
            ax.legend()
            st.pyplot(fig)
            plt.close()

    # ─── Forecast หลายเดือน ───
    st.markdown("---")
    st.subheader("📅 ทำนายหลายเดือนล่วงหน้า")
    n_months = st.slider("จำนวนเดือนที่ต้องการทำนาย", 1, 24, 12)

    if st.button("🔭 ทำนาย Future Forecast", use_container_width=True):
        if not model_loaded:
            st.error("กรุณา run notebook ก่อน")
        else:
            monthly_ts = df.groupby(['year','month']).size().reset_index(name='count')
            monthly_ts = monthly_ts[(monthly_ts['year'] >= 1990)].sort_values(['year','month'])
            history    = list(monthly_ts['count'].values)

            forecast_dates, forecast_vals = [], []
            cur_year, cur_month = 2015, 1

            for _ in range(n_months):
                X_in  = build_features(cur_year, cur_month, pd.Series(history))
                p     = max(0, int(round(model.predict(X_in)[0])))
                forecast_dates.append(pd.Timestamp(year=cur_year, month=cur_month, day=1))
                forecast_vals.append(p)
                history.append(p)
                cur_month += 1
                if cur_month > 12:
                    cur_month = 1
                    cur_year += 1

            forecast_df = pd.DataFrame({'date': forecast_dates, 'predicted': forecast_vals})

            fig, ax = plt.subplots(figsize=(13, 4))
            hist_plot = monthly_ts[monthly_ts['year'] >= 2010].copy()
            hist_plot['date'] = pd.to_datetime(hist_plot[['year','month']].assign(day=1))
            ax.plot(hist_plot['date'], hist_plot['count'], color='steelblue', label='ข้อมูลจริง')
            ax.plot(forecast_df['date'], forecast_df['predicted'], color='tomato', linestyle='--', marker='o', markersize=4, label='ทำนาย')
            ax.axvline(pd.Timestamp('2015-01-01'), color='gray', linestyle=':', alpha=0.6)
            ax.set_title(f'UFO Forecast: {n_months} เดือน')
            ax.set_xlabel('เดือน')
            ax.set_ylabel('จำนวนการพบ UFO')
            ax.legend()
            st.pyplot(fig)
            plt.close()

            st.dataframe(forecast_df.rename(columns={'date':'เดือน','predicted':'ทำนาย (ครั้ง)'}), use_container_width=True)

# ─── Page 2: Feature Importance ────────────────────────────────────────────────
elif page == "📊 Feature Importance":
    st.title("📊 Feature Importance (โบนัส)")
    st.markdown("แสดงว่า features ใดมีผลต่อการทำนายมากที่สุด")

    if not model_loaded:
        st.error("กรุณา run notebook ก่อน")
    else:
        try:
            # ใช้ RF model เสมอ เพราะมี feature_importances_ แน่นอน
            inner_model = rf_model.named_steps['model']
        except Exception:
            inner_model = rf_model

        if hasattr(inner_model, 'feature_importances_'):
            fi = pd.Series(inner_model.feature_importances_, index=features).sort_values(ascending=False)

            feature_desc = {
                't':          '📈 Trend — ทิศทางขาขึ้นตามเวลา',
                'lag_1':      '⏱️ Lag 1 — ค่าเดือนก่อนหน้า',
                'lag_2':      '⏱️ Lag 2 — ค่า 2 เดือนก่อน',
                'lag_12':     '📅 Lag 12 — ค่าเดือนเดียวกันปีที่แล้ว',
                'rolling_3':  '📊 Rolling 3 — ค่าเฉลี่ย 3 เดือน',
                'rolling_12': '📊 Rolling 12 — ค่าเฉลี่ย 12 เดือน',
                'month_sin':  '🔄 Month Sin — Seasonality (วงรอบเดือน)',
                'month_cos':  '🔄 Month Cos — Seasonality (วงรอบเดือน)',
            }

            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(9, 5))
                colors_fi = cm.viridis(np.linspace(0.2, 0.85, len(fi)))
                bars = ax.barh(fi.index[::-1], fi.values[::-1], color=colors_fi[::-1])
                for bar, val in zip(bars, fi.values[::-1]):
                    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                            f'{val:.3f}', va='center', fontsize=10)
                ax.set_title(f'Feature Importance — {meta["best_model"]}', fontsize=13)
                ax.set_xlabel('Importance Score')
                ax.set_xlim(0, fi.max() * 1.2)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader("คำอธิบาย Features")
                for feat, imp in fi.items():
                    pct = imp / fi.sum() * 100
                    st.markdown(f"**{feature_desc.get(feat, feat)}**")
                    st.progress(float(imp / fi.max()))
                    st.caption(f"Importance: {imp:.3f} ({pct:.1f}%)")
                    st.markdown("")

            # Interactive: what-if
            st.markdown("---")
            st.subheader("🔍 ความสำคัญของ Lag Features")
            st.markdown("""
            - **lag_1** (ค่าเดือนที่แล้ว) มักมี importance สูงที่สุด → พฤติกรรมการรายงาน UFO มี autocorrelation สูง
            - **rolling_12** (ค่าเฉลี่ย 12 เดือน) สะท้อน trend ระยะยาว
            - **lag_12** (ปีที่แล้ว เดือนเดียวกัน) จับ seasonal pattern ประจำปี
            - **t** (trend) สะท้อนการเพิ่มขึ้นของการรายงานตามยุค internet
            """)
        else:
            st.warning("โมเดลที่เลือกไม่สามารถแสดง feature importance ได้ (ต้องเป็น tree-based model)")

# ─── Page 3: EDA ───────────────────────────────────────────────────────────────
elif page == "📈 ข้อมูล EDA":
    st.title("📈 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Time Series", "Seasonality", "Distribution"])

    with tab1:
        st.subheader("จำนวนการพบ UFO รายปี")
        yearly = df.groupby('year').size().reset_index(name='count')
        yearly = yearly[(yearly['year'] >= 1940) & (yearly['year'] <= 2014)]
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.fill_between(yearly['year'], yearly['count'], alpha=0.3, color='steelblue')
        ax.plot(yearly['year'], yearly['count'], color='steelblue', linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('Sightings')
        ax.set_title('UFO Sightings per Year (1940–2014)')
        st.pyplot(fig)
        plt.close()
        st.info("💡 การพบ UFO เพิ่มขึ้นอย่างชัดเจนตั้งแต่ปี 1990 ตรงกับยุคอินเทอร์เน็ต ทำให้คนรายงานได้ง่ายขึ้น")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Seasonality รายเดือน")
            monthly_avg = df.groupby('month').size()
            month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(month_names, monthly_avg.values, color='mediumorchid', edgecolor='white')
            ax.set_title('Sightings by Month')
            ax.set_ylabel('Total Sightings')
            st.pyplot(fig)
            plt.close()
            st.info("💡 พบมากที่สุดในเดือนกรกฎาคม (วัน 4th of July สหรัฐ)")

        with col2:
            st.subheader("Sightings by Hour of Day")
            hourly = df.groupby(df['datetime'].dt.hour).size()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(hourly.index, hourly.values, marker='o', color='tomato', linewidth=2)
            ax.fill_between(hourly.index, hourly.values, alpha=0.2, color='tomato')
            ax.set_xlabel('Hour')
            ax.set_title('Sightings by Hour')
            ax.set_xticks(range(0, 24, 2))
            st.pyplot(fig)
            plt.close()
            st.info("💡 พบมากที่สุดช่วง 20:00–22:00 — คืนที่ท้องฟ้ามืดและคนยังตื่นอยู่")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 UFO Shapes")
            top_shapes = df['shape'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(top_shapes.index[::-1], top_shapes.values[::-1], color='steelblue')
            ax.set_xlabel('Count')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Sightings by Country")
            top_country = df['country'].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(top_country.index, top_country.values, color='coral', edgecolor='white')
            ax.set_title('Top Countries')
            plt.xticks(rotation=30)
            st.pyplot(fig)
            plt.close()

        st.subheader("📊 สถิติพื้นฐาน")
        st.dataframe(df[['duration (seconds)','latitude','longitude ']].describe().round(2), use_container_width=True)
