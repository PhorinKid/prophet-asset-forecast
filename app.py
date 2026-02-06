import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import shutil
import os

# 모듈들 가져오기
from src.db import load_data, load_merged_data
from src.processing import preprocess_data
from src.models import ModelFactory

# -------------------------------------------------------------------------
# 호환용 상수 설정
# -------------------------------------------------------------------------
TIME_STEP_MINUTES = 30
POINTS_PER_DAY = int(24 * 60 / TIME_STEP_MINUTES)

# -------------------------------------------------------------------------
# 0. 페이지 설정
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Prophet - 디지털 자산 시세 변동 예측 모델",
    layout="wide"
)

# 세션 초기화
if "item_results" not in st.session_state:
    st.session_state.item_results = {}

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

st.title("디지털 자산 시세 변동 예측 모델")
st.caption("AI 앙상블 모델(LGBM, XGBoost, NeuralProphet)을 활용하여 시세를 분석합니다.")

# -------------------------------------------------------------------------
# 1. 사이드바 - 검색 / 학습 설정 (팀원 코드 통합)
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("아이템 검색")

    # 1. 데이터 로드 (DB에서 전체 아이템 목록 가져오기)
    df_meta = load_merged_data()
    
    # 등급 정렬 기준
    grade_order = ["에스더", "고대", "유물", "전설", "영웅", "희귀", "고급", "일반"]

    df_meta['grade'] = pd.Categorical(
        df_meta['grade'],
        categories=grade_order,
        ordered=True
    )

    # 정렬 (이름순 -> 등급순)
    df_meta = df_meta.sort_values(by=['name', 'grade'])

    # 2. 아이템 이름 리스트 추출 (중복 제거)
    unique_names = df_meta['name'].unique()

    default_target = "운명의 파괴석"
    names_list = list(unique_names)

    if default_target in names_list:
        default_index = names_list.index(default_target)
    else:
        default_index = 0

    # 3. [1단계] 아이템 이름 선택
    selected_name = st.selectbox(
        "분석 대상 아이템",
        unique_names,
        index=default_index
    )

    # 4. [2단계] 선택된 아이템에 맞는 등급 리스트 생성
    # 선택된 이름의 데이터만 뽑음
    item_rows = df_meta[df_meta['name'] == selected_name]

    # 해당 아이템이 가진 등급들만 추출
    grades_for_item = item_rows.sort_values('grade')['grade'].unique()
    grade_options = list(grades_for_item)

    # (예외처리) 만약 데이터가 꼬여서 등급이 하나도 없으면?
    if len(grade_options) == 0:
        st.error("해당 아이템의 등급 정보가 없습니다.")
        target_grade = None
    else:
        target_grade = st.selectbox(
            f"'{selected_name}' 등급",
            grade_options,
            index=0
        )

    st.markdown("---")
    days_to_show = st.slider("그래프 표시 기간 (일)", 1, 14, 3)

    st.markdown("---")
    if target_grade:
        run_button = st.button("AI 예측 시작", type="primary", use_container_width=True)
    else:
        run_button = False
        st.warning("등급을 선택해야 예측할 수 있습니다.")

    # 관리자 설정
    st.markdown("---")
    with st.expander("⚙️ 관리자 설정"):
        clear_model_button = st.button("모델 초기화")
        if clear_model_button:
            model_path = "models"
            if os.path.exists(model_path):
                try:
                    shutil.rmtree(model_path)
                    os.makedirs(model_path)
                    st.success("모델 삭제 완료!")
                    st.session_state.item_results = {}
                    st.session_state.forecast_result = None
                except Exception as e:
                    st.error(f"실패: {e}")

# -------------------------------------------------------------------------
# 2. 메인 로직
# -------------------------------------------------------------------------
if run_button:
    session_key = f"{selected_name} [{target_grade}]"

    if session_key in st.session_state.item_results:
        st.success(f"'{session_key}'의 이전 분석 결과를 불러옵니다.")
        st.session_state.forecast_result = st.session_state.item_results[session_key]
    else:
        with st.spinner(f"[{session_key}] 신규 분석을 시작합니다."):
            df_prices, df_notices, item_id = load_data(selected_name, target_grade)

            if item_id is None or df_prices is None or df_prices.empty:
                st.error("해당 아이템의 데이터를 찾을 수 없습니다.")
            else:
                df_ml_now = preprocess_data(df_prices, df_notices)
                TRAIN_CUTOFF = pd.Timestamp("2026-02-04 06:00:00")
                df_ml_train = df_ml_now[df_ml_now.index < TRAIN_CUTOFF]
                factory = ModelFactory()
                try:
                    factory.load_models(item_id)
                except:
                    status_container = st.empty()
                    status_container.warning("신규 학습 진행 중... (1~2분 소요)")
                    factory.train_all(df_ml_train)
                    factory.save_models(item_id)
                    status_container.empty()
                
                forecast = factory.predict_future_3days(df_ml_now, item_id, df_notices)
                
                new_result = {
                    "item_name": selected_name,
                    "df_prices": df_prices,
                    "forecast": forecast,
                    "current_price": int(df_prices['current_min_price'].iloc[-1])
                }
                
                st.session_state.item_results[session_key] = new_result
                st.session_state.forecast_result = new_result
                st.success(f"[{session_key}] 분석 완료")

# -------------------------------------------------------------------------
# 3. 결과 화면 표시 (슬라이더 연동)
# -------------------------------------------------------------------------
if st.session_state.forecast_result:
    res = st.session_state.forecast_result
    item_name = res["item_name"]
    forecast = res["forecast"]
    df_prices = res["df_prices"]
    curr_p = res["current_price"]

    # 데이터 병합
    past_df = df_prices[['logged_at', 'current_min_price']].copy()
    past_df.columns = ['ds', 'Price']
    past_df['Model'] = 'Actual'
    past_df['Opacity'] = 1.0
    past_df['StrokeWidth'] = 1

    future_melted = forecast.melt(id_vars=['ds'], value_vars=['forecast', 'lgbm', 'xgb', 'np'], var_name='Model', value_name='Price')
    future_melted['Opacity'] = future_melted['Model'].apply(lambda x: 1.0 if x == 'forecast' else 0.5)
    future_melted['StrokeWidth'] = future_melted['Model'].apply(lambda x: 1.2 if x == 'forecast' else 1)

    full_df = pd.concat([past_df, future_melted], ignore_index=True)
    full_df['ds'] = pd.to_datetime(full_df['ds'])

    # 🚨 [핵심 변경] 슬라이더(days_to_show) 값에 따라 그래프 범위 자동 조절
    last_predict_date = future_melted['ds'].max()

    # "최근 예측 기간" 슬라이더 값만큼 과거를 보여줍니다.
    view_start = last_predict_date - pd.Timedelta(days=days_to_show)
    view_end = last_predict_date + pd.Timedelta(hours=0)

    st.markdown(f"### {item_name} 모델별 상세 예측 트렌드")

    y_min = int(full_df['Price'].min() * 0.98)
    y_max = int(full_df['Price'].max() * 1.02)

    lines = alt.Chart(full_df).mark_line().encode(
        x=alt.X('ds:T', title='날짜 및 시간',
                axis=alt.Axis(format='%m/%d %H:%M', tickCount=8),
                scale=alt.Scale(domain=[view_start, view_end])),
        y=alt.Y('Price:Q', title='가격 (Gold)', scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.Color('Model:N', scale=alt.Scale(
            domain=['Actual', 'forecast', 'lgbm', 'xgb', 'np'],
            range=['#808080', '#FF4B4B', '#1C83E1', '#00C781', '#FFAA00']
        ), title="모델 구분"),
        opacity=alt.Opacity('Opacity:Q', legend=None),
        strokeWidth=alt.StrokeWidth('StrokeWidth:Q', legend=None),
        tooltip=['ds:T', 'Model:N', 'Price:Q']
    ).interactive()

    # 수요일 가이드라인
    wednesdays = pd.date_range(start=full_df['ds'].min(), end=full_df['ds'].max(), freq='W-WED').normalize() + pd.Timedelta(hours=6)
    rules = alt.Chart(pd.DataFrame({'ds': wednesdays})).mark_rule(color='gold', strokeDash=[5, 5]).encode(x='ds:T')

    # 지표 표시
    future_p = int(forecast['forecast'].iloc[-1])
    diff = future_p - curr_p
    diff_percent = (diff / curr_p) * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("현재 시세", f"{curr_p:,.0f} G")
    m2.metric("3일 뒤 예상", f"{future_p:,.0f} G", delta=f"{diff:,.0f} G ({diff_percent:.1f}%)")
    m3.metric("기간 설정", f"{days_to_show}일 보기", help="사이드바 슬라이더로 조절 가능")

    # 그래프 표시
    st.altair_chart((lines + rules), use_container_width=True)
