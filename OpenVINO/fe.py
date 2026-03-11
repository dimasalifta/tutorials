import streamlit as st

st.title("MediaMTX HLS Viewer")

# HLS dari MediaMTX
st.video("http://localhost:8889/cam1/index.m3u8")