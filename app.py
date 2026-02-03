import streamlit as st
import socket

# 페이지 제목
st.set_page_config(page_title="배포 테스트", page_icon="🚀")

st.title("🚀 배포 성공! 연결 완료!")

st.header("포린님의 EC2가 정상 작동 중입니다.")
st.write("GitHub Actions -> Docker Hub -> EC2 연결이 완벽합니다.")

# 현재 실행 중인 컨테이너 정보 (확인용)
hostname = socket.gethostname()
st.info(f"현재 실행 중인 컨테이너 ID: {hostname}")

st.balloons()
