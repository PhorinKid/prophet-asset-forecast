import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def get_ai_advice(item_name, current_price, df_forecast):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ .env 파일에 OPENAI_API_KEY가 설정되지 않았습니다."

    client = OpenAI(api_key=api_key)

    # -----------------------------------------------------------
    # 1. [Python] 절대 팩트 계산
    # -----------------------------------------------------------
    
    # 최고점 (Max)
    idx_max = df_forecast['forecast'].idxmax()
    max_val = int(df_forecast.loc[idx_max]['forecast'])
    max_time_exact = df_forecast.loc[idx_max]['ds'].strftime("%m/%d %H:%M")
    
    # 최저점 (Min)
    idx_min = df_forecast['forecast'].idxmin()
    min_val = int(df_forecast.loc[idx_min]['forecast'])
    min_time_exact = df_forecast.loc[idx_min]['ds'].strftime("%m/%d %H:%M")

    # -----------------------------------------------------------
    # 2. [Data] 전체 데이터 전송
    # -----------------------------------------------------------
    df_full = df_forecast[['ds', 'forecast']].copy()
    data_str = df_full.to_string(index=False)

    prompt = f"""
    너는 노련한 로스트아크 투자 전문가야. '{item_name}'의 향후 3일(30분 봉) 시세 데이터를 분석해줘.

    [절대 팩트 (참고용)]
    - 현재가: {current_price} G
    - 데이터상 최저점: {min_val} G (찍은 시각: {min_time_exact})
    - 데이터상 최고점: {max_val} G (찍은 시각: {max_time_exact})

    [향후 3일(30분 봉) 시세 예측 데이터]
    {data_str}

    [분석 요청]
    위 데이터를 보고, 사용자가 실제로 수익을 낼 수 있는 "유효 타격 시간대"와 "안전 매매가"를 판단해줘.
    *주의: 단순히 팩트 수치를 그대로 베끼지 말고, 데이터 흐름(급등/횡보)을 보고 사람이 대응 가능한 시간 범위를 설정할 것.*

    [출력 양식]
    1. **현재 가격**: {current_price} G
    2. **최고점 예상**: {max_val} G 부근
        -> 예상 구간: (AI가 데이터 흐름을 보고 '00일 00시~00시' 처럼 판단해서 작성)
    3. **최저점 예상**: {min_val} G 부근
        -> 예상 구간: (AI가 데이터 흐름을 보고 '00일 00시~00시' 처럼 판단해서 작성)
    4. **추천 구매가**: **0000 G** 이하
        -> (전략: 하락 추세의 기울기를 보고, 체결 가능한 안전한 가격 산정)
    5. **추천 판매가**: **0000 G** 이상
        -> (전략: 상승 추세의 힘을 보고, 욕심부리지 않고 팔릴 가격 산정)
    
    **요약:** (한 줄로 매매 포지션 추천)
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "데이터의 변동성을 해석하여 실질적인 조언을 주는 투자 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"🤖 AI 분석 실패: {str(e)}"
