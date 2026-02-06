import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import shutil
import os

from src.db import load_data, load_merged_data
from src.processing import preprocess_data
from src.models import ModelFactory
from src.ai_advisor import get_ai_advice

# -------------------------------------------------------------------------
# 설정 및 상수
# -------------------------------------------------------------------------
TIME_STEP_MINUTES = 30
POINTS_PER_DAY = int(24 * 60 / TIME_STEP_MINUTES)

st.set_page_config(
    page_title="Digital Asset Forecast",
    layout="wide"
)

# 세션 상태 초기화
if "item_results" not in st.session_state:
    st.session_state.item_results = {}

if "ai_advice_cache" not in st.session_state:
    st.session_state.ai_advice_cache = {}

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

st.title("디지털 자산 시세 변동 예측 모델")
st.caption("AI 앙상블 모델(LGBM, XGBoost, NeuralProphet) 기반 시세 분석")

# -------------------------------------------------------------------------
# 사이드바: 설정 및 검색
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("아이템 검색")

    df_meta = load_merged_data()
    
    grade_order = ["에스더", "고대", "유물", "전설", "영웅", "희귀", "고급", "일반"]
    df_meta['grade'] = pd.Categorical(
        df_meta['grade'], categories=grade_order, ordered=True
    )
    df_meta = df_meta.sort_values(by=['name', 'grade'])

    unique_names = df_meta['name'].unique()
    default_target = "운명의 파괴석"
    
    default_index = list(unique_names).index(default_target) if default_target in unique_names else 0

    selected_name = st.selectbox("분석 대상 아이템", unique_names, index=default_index)

    item_rows = df_meta[df_meta['name'] == selected_name]
    grade_options = list(item_rows.sort_values('grade')['grade'].unique())

    if not grade_options:
        st.error("등급 정보가 없습니다.")
        target_grade = None
    else:
        target_grade = st.selectbox(f"'{selected_name}' 등급", grade_options, index=0)

    st.markdown("---")
    days_to_show = st.slider("그래프 표시 기간 (일)", 1, 14, 3)
    use_global_scale = st.checkbox("Y축 범위를 전체 기간으로 고정", value=False)

    st.markdown("---")
    run_button = st.button("AI 예측 시작", type="primary", use_container_width=True) if target_grade else False

    st.markdown("---")
    with st.expander("⚙️ 관리자 설정"):
        if st.button("모델 초기화"):
            model_path = "models"
            if os.path.exists(model_path):
                try:
                    shutil.rmtree(model_path)
                    os.makedirs(model_path)
                    st.session_state.item_results = {}
                    st.session_state.forecast_result = None
                    st.success("모델 캐시 삭제 완료")
                except Exception as e:
                    st.error(f"실패: {e}")

# -------------------------------------------------------------------------
# 메인 로직: 데이터 분석 및 예측
# -------------------------------------------------------------------------
if run_button:
    session_key = f"{selected_name} [{target_grade}]"

    if session_key in st.session_state.item_results:
        st.session_state.forecast_result = st.session_state.item_results[session_key]
    else:
        with st.spinner(f"[{session_key}] 데이터 분석 및 예측 수행 중..."):
            df_prices, df_notices, item_id = load_data(selected_name, target_grade)

            if item_id is None or df_prices is None or df_prices.empty:
                st.error("데이터를 찾을 수 없습니다.")
            else:
                df_ml_now = preprocess_data(df_prices, df_notices)
                
                # [발표용 하드코딩]
                # 발표 시연을 위해 특정 시점으로 고정합니다.
                # (추후 매주 수요일 리셋 시간에 맞춰 자동 갱신되도록 변경 예정)
                TRAIN_CUTOFF = pd.Timestamp("2026-02-04 06:00:00")
                
                df_ml_train = df_ml_now[df_ml_now.index < TRAIN_CUTOFF]
                
                factory = ModelFactory()
                try:
                    factory.load_models(item_id)
                except:
                    # 모델이 없으면 즉석 학습
                    status_msg = st.empty()
                    status_msg.info("신규 모델 학습 진행 중...")
                    factory.train_all(df_ml_train)
                    factory.save_models(item_id)
                    status_msg.empty()
                
                forecast = factory.predict_future_3days(df_ml_now, item_id, df_notices)
                
                new_result = {
                    "item_name": selected_name,
                    "df_prices": df_prices,
                    "forecast": forecast,
                    "current_price": int(df_prices['current_min_price'].iloc[-1])
                }
                
                st.session_state.item_results[session_key] = new_result
                st.session_state.forecast_result = new_result

# -------------------------------------------------------------------------
# 결과 시각화
# -------------------------------------------------------------------------
if st.session_state.forecast_result:
    res = st.session_state.forecast_result
    item_name = res["item_name"]
    forecast = res["forecast"]
    df_prices = res["df_prices"]
    curr_p = res["current_price"]

    # 1. 데이터 가공
    past_df = df_prices[['logged_at', 'current_min_price']].copy()
    past_df.columns = ['ds', 'Price']
    past_df['Model'] = 'Actual'
    past_df['Opacity'] = 1.0
    past_df['StrokeWidth'] = 1

    future_melted = forecast.melt(id_vars=['ds'], value_vars=['forecast', 'lgbm', 'xgb', 'nural_prophet'], var_name='Model', value_name='Price')
    future_melted['Opacity'] = future_melted['Model'].apply(lambda x: 1.0 if x == 'forecast' else 0.7)
    future_melted['StrokeWidth'] = future_melted['Model'].apply(lambda x: 1.2 if x == 'forecast' else 1)

    full_df = pd.concat([past_df, future_melted], ignore_index=True)
    full_df['ds'] = pd.to_datetime(full_df['ds'])

    # 2. 뷰 범위 설정
    last_predict_date = future_melted['ds'].max()
    view_start = last_predict_date - pd.Timedelta(days=days_to_show)
    view_end = last_predict_date + pd.Timedelta(minutes=TIME_STEP_MINUTES * 4)

    # 3. Y축 스케일 계산
    if use_global_scale:
        y_min_val = full_df['Price'].min()
        y_max_val = full_df['Price'].max()
    else:
        visible_df = full_df[full_df['ds'] >= view_start]
        if not visible_df.empty:
            y_min_val = visible_df['Price'].min()
            y_max_val = visible_df['Price'].max()
        else:
            y_min_val = full_df['Price'].min()
            y_max_val = full_df['Price'].max()

    margin = (y_max_val - y_min_val) * 0.1
    y_min = int(max(0, y_min_val - margin))
    y_max = int(y_max_val + margin)

    # 4. 차트 생성
    lines = alt.Chart(full_df).mark_line().encode(
        x=alt.X('ds:T', title='날짜 및 시간',
                axis=alt.Axis(format='%m/%d %H:%M', tickCount=8),
                scale=alt.Scale(domain=[view_start, view_end])),
        y=alt.Y('Price:Q', title='가격 (Gold)',
                scale=alt.Scale(domain=[y_min, y_max], zero=False)),
        color=alt.Color('Model:N', scale=alt.Scale(
            domain=['Actual', 'forecast', 'lgbm', 'xgb', 'nural_prophet'],
            range=['#808080', '#FF4B4B', '#1C83E1', '#00C781', '#FFAA00']
        ), title="모델"),
        opacity=alt.Opacity('Opacity:Q', legend=None),
        strokeWidth=alt.StrokeWidth('StrokeWidth:Q', legend=None),
        tooltip=['ds:T', 'Model:N', 'Price:Q']
    ).interactive(bind_y=False)

    wednesdays = pd.date_range(start=view_start, end=view_end, freq='W-WED').normalize() + pd.Timedelta(hours=6)
    rules = alt.Chart(pd.DataFrame({'ds': wednesdays})).mark_rule(color='gold', strokeDash=[5, 5]).encode(x='ds:T')

    # 5. AI 가이드 (캐싱 적용)
    future_vals = forecast['forecast'].values
    min_pred = int(np.min(future_vals))
    max_pred = int(np.max(future_vals))
    
    if item_name not in st.session_state.ai_advice_cache:
        with st.spinner(f"AI 전략 분석 중..."):
            advice_text = get_ai_advice(item_name, curr_p, forecast)
            st.session_state.ai_advice_cache[item_name] = advice_text
    
    cached_advice = st.session_state.ai_advice_cache[item_name]
    
    # 6. 화면 출력
    st.subheader("AI 투자 전략 가이드")
    st.info(cached_advice, icon="📊")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("현재 시세", f"{curr_p:,.0f} G")
    c2.metric("예측 최저", f"{min_pred:,.0f} G", delta=f"{min_pred - curr_p:,.0f} G", delta_color="inverse")
    c3.metric("예측 최고", f"{max_pred:,.0f} G", delta=f"{max_pred - curr_p:,.0f} G")
    
    st.caption("※ 예측 데이터(3일) 기반 분석 결과입니다.")

    st.subheader(f"{item_name} 가격 예측 트렌드")
    st.altair_chart((lines + rules), use_container_width=True)
