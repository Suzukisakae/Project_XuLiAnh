import streamlit as st
from streamlit_option_menu import option_menu

from Face_Detect import nhan_dang_khuon_mat
from Fruits_Detect import nhan_dang_trai_cay
from Image_Processing import xu_li_anh
from MNIST_Detect import chu_so_viet_tay
from Welcome import Welcome
from Style_Transfer import style_transfer
# streamlit run pages/main.py

with st.sidebar:
    selected = option_menu("Menu", ['Trang Chủ', 'Nhận diện khuôn mặt', 'Nhận diện trái cây', 'Chữ số viết tay', 'Xử lý ảnh', 'Đổi phong cách'], 
        icons=[], menu_icon="cast", default_index=0)
if selected == 'Trang Chủ':
    st.title("Đồ án Xử lí ảnh")
    Welcome()
elif selected == 'Nhận diện khuôn mặt':
    st.title("Nhận diện khuôn mặt")
    nhan_dang_khuon_mat()

elif selected == 'Nhận diện trái cây':
    st.title("Nhận diện trái cây")
    nhan_dang_trai_cay()
elif selected == 'Xử lý ảnh':
    st.title("Xử lý ảnh")
    xu_li_anh()
elif selected == 'Chữ số viết tay':
    st.title("Nhận diện chữ số viết tay")
    chu_so_viet_tay()
elif selected == 'Đổi phong cách':
    style_transfer()
