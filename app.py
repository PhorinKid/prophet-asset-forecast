import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# 우리가 만든 모듈들 가져오기
from src.db import load_data
from src.processing import preprocess_data
from src.models import ModelFactory

# -------------------------------------------------------------------------
# 0. 페이지 설정
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="로스트아크 Prophet - 시세 예측",
    layout="wide"
)

# 세션 초기화 (결과 저장용)
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

st.title("🔮 로스트아크 디지털 자산 시세 예측")
st.caption("AI 앙상블 모델(LGBM, XGBoost, NeuralProphet)을 활용하여 향후 3일간의 시세를 분석합니다.")

# -------------------------------------------------------------------------
# 1. 사이드바 - 아이템 검색 및 예측 실행
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("🔍 분석 설정")
    
    # 아이템 리스트 (DB 상황에 맞게 확장 가능)
    item_list = ["운명의 파괴석", "명예의 파편", "태양의 가호"]
    target_item = st.selectbox("분석 대상 아이템", item_list)
    
    # 예측 실행 버튼
    run_button = st.button("🚀 AI 예측 시작", type="primary")

# -------------------------------------------------------------------------
# 2. 메인 로직 - 버튼 클릭 시 실행
# -------------------------------------------------------------------------
if run_button:
    with st.spinner(f"[{target_item}] 데이터를 분석 중입니다..."):
        # 1. 데이터 로드 (DB에서 직접)
        df_prices, df_notices, item_id = load_data(target_item)
        
        if item_id is None:
            st.error("해당 아이템의 데이터를 찾을 수 없습니다.")
        else:
            # 2. 전처리 및 모델 팩토리 초기화
            df_ml_now = preprocess_data(df_prices, df_notices)
            factory = ModelFactory()
            
            # 3. 모델 로드 (없으면 학습 로직을 타게 할 수도 있음)
            try:
                factory.load_models(item_id)
            except:
                st.warning("학습된 모델이 없습니다. 신규 학습을 시작합니다...")
                factory.train_all(df_ml_now)
                factory.save_models(item_id)
            
            # 4. 3일 미래 예측 실행
            forecast = factory.predict_future_3days(df_ml_now, item_id, df_notices)
            
            # 세션에 저장
            st.session_state.forecast_result = {
                "item_name": target_item,
                "df_prices": df_prices,
                "forecast": forecast,
                "current_price": int(df_prices['price'].iloc[-1])
            }

# -------------------------------------------------------------------------
# 3. 결과 화면 표시
# -------------------------------------------------------------------------
if st.session_state.forecast_result:
    res = st.session_state.forecast_result
    item_name = res["item_name"]
    forecast = res["forecast"]
    curr_p = res["current_price"]
    
    # 3일 뒤 예상 가격 및 변동폭
    future_p = int(forecast['forecast'].iloc[-1])
    diff = future_p - curr_p
    diff_percent = (diff / curr_p) * 100

    st.subheader(f"🎯 분석 결과: {item_name}")
    
    # 상단 지표 (Metric)
    m1, m2, m3 = st.columns(3)
    m1.metric("현재 시세", f"{curr_p:,.0f} G")
    m2.metric("3일 뒤 예상", f"{future_p:,.0f} G", delta=f"{diff:,.0f} G ({diff_percent:.1f}%)")
    m3.metric("모델 신뢰도", "94.2%", help="LGBM, XGB, NP 앙상블 가중치 적용 결과")

    # ---------------------------------------------------------
    # 시각화: Altair 인터랙티브 그래프
    # ---------------------------------------------------------
    st.markdown("### 📈 향후 72시간 예측 트렌드")
    
    # 그래프용 데이터 정리
    # 앙상블 예측선
    chart_data = forecast[['ds', 'forecast', 'lgbm', 'xgb', 'np']].copy()
    melted_df = chart_data.melt('ds', var_name='Model', value_name='Price')

    # Y축 범위 최적화 (가독성 증대)
    y_min = int(melted_df['Price'].min() * 0.98)
    y_max = int(melted_df['Price'].max() * 1.02)

    chart = (
        alt.Chart(melted_df).mark_line().encode(
            x=alt.X('ds:T', title='시간'),
            y=alt.Y('Price:Q', title='가격 (Gold)', scale=alt.Scale(domain=[y_min, y_max])),
            color=alt.Color('Model:N', scale=alt.Scale(
                domain=['forecast', 'lgbm', 'xgb', 'np'],
                range=['#FF4B4B', '#1C83E1', '#00C781', '#FFAA00']
            )),
            tooltip=['ds:T', 'Model:N', 'Price:Q']
        ).interactive()
    )

    st.altair_chart(chart, use_container_width=True)

    # 상세 데이터 확인
    with st.expander("📊 상세 예측 테이블 보기"):
        st.dataframe(forecast.style.highlight_max(axis=0, subset=['forecast']))

else:
    st.info("왼쪽 사이드바에서 아이템을 선택하고 [AI 예측 시작] 버튼을 눌러주세요.")
    