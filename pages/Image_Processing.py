import streamlit as st
from PIL import Image
import numpy as np
import cv2
from xu_ly_anh import Negative, Logarit, Power, PiecewiseLinear, Histogram, LocalHist, HistStat, MyFilter2D, MySmooth, OnSharpen, Gradient
from xu_ly_anh import Spectrum, HighpassFilter, DrawNotchRejectFilter, RemoveMoire
from xu_ly_anh import CreateMotionNoise, DenoiseMotion
from xu_ly_anh import ConnectedComponent, CountRice

def xu_li_anh():
    with st.sidebar:
        st.title("Xử lý ảnh")
        selected_process = st.selectbox("Phương pháp xử lý ảnh", ["","Biến đổi độ sáng và lọc trong không gian",
                                                         "Lọc trong miền tần số", "Khôi phục ảnh", "Xử lý ảnh hình thái"])
    if selected_process == "":
        st.info("Vui lòng chọn một phương pháp xử lý ảnh trong menu bên trái")
    if selected_process == "Biến đổi độ sáng và lọc trong không gian":
        xu_li_anh_c3()
    if selected_process == "Lọc trong miền tần số":
        xu_li_anh_c4()
    if selected_process == "Khôi phục ảnh":
        xu_li_anh_c5()
    if selected_process == "Xử lý ảnh hình thái":
        xu_li_anh_c9()

def xu_li_anh_c3():
    st.header("Biến đổi độ sáng và lọc trong không gian")
    imgin = None
    imgout = None
    
    file_uploaded = st.file_uploader("Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'tif'])
    if file_uploaded is not None:
        file_bytes = np.asarray(bytearray(file_uploaded.read()), dtype=np.uint8)
        imgin = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(imgin, channels="GRAY", use_column_width=True, caption="Input Image", width=100)
        st.write("Kích thước ảnh: ", imgin.shape)
    operation = st.selectbox("Các tùy chọn", ["", "Negative", "Logarit", "Power",
                                                            "PiecewiseLinear", "Histogram", "Hist Equal",
                                                            "Local Hist", "Hist Stat", "Filter 2D", "My Smooth",
                                                            "Smooth Box", "Smooth Gaussian",
                                                            "Median filter", "Sharpen", "Gradient"])
    if imgin is not None:
        transfer = st.button("Chuyển đổi")
        if transfer:
            if operation == "Negative":
                imgout = Negative(imgin)
            elif operation == "Logarit":
                imgout = Logarit(imgin)
            elif operation == "Power":
                imgout = Power(imgin)
            elif operation == "PiecewiseLinear":
                imgout = PiecewiseLinear(imgin)
            elif operation == "Histogram":
                imgout = Histogram(imgin)
            elif operation == "Hist Equal":
                imgout = cv2.equalizeHist(imgin)
            elif operation == "Local Hist":
                imgout = LocalHist(imgin)
            elif operation == "Hist Stat":
                imgout = HistStat(imgin)
            elif operation == "Filter 2D":
                imgout = MyFilter2D(imgin)
            elif operation == "My Smooth":
                imgout = MySmooth(imgin)
            elif operation == "Smooth Box":
                imgout = cv2.blur(imgin, (5, 5))
            elif operation == "Smooth Gaussian":
                imgout = cv2.GaussianBlur(imgin, (5, 5), 0)
            elif operation == "Median filter":
                imgout = cv2.medianBlur(imgin, 5)
            elif operation == "Sharpen":
                imgout = OnSharpen(imgin)
            elif operation == "Gradient":
                imgout = Gradient(imgin)
            st.info("Ảnh sau khi xử lí")
            st.image(imgout, channels="GRAY", use_column_width=True, caption="Output Image", width=100)
            st.write("Kích thước ảnh: ", imgout.shape)
    else:
        st.warning("Vui lòng tải ảnh lên")

def xu_li_anh_c4():
    st.header("Lọc trong miền tần số")
    imgin = None
    imgout = None
    file_uploaded = st.file_uploader("Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'tif'])
    if file_uploaded is not None:
        file_bytes = np.asarray(bytearray(file_uploaded.read()), dtype=np.uint8)
        imgin = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(imgin, channels="GRAY", use_column_width=True, caption="Input Image", width=100)
        st.write("Kích thước ảnh: ", imgin.shape)
    operation = st.selectbox("Các tùy chọn", ["", "Spectrum", "High pass filter", "Notch Reject", "Remove Moire"])
    if imgin is not None:
        transfer = st.button("Chuyển đổi")
        if transfer:
            if operation == "Spectrum":
                imgout = Spectrum(imgin)
            elif operation == "High pass filter":
                imgout = HighpassFilter(imgin)
            elif operation == "Notch Reject":
                imgout = DrawNotchRejectFilter()
            elif operation == "Remove Moire":
                imgout = RemoveMoire(imgin)
            st.info("Ảnh sau khi xử lí")
            st.image(imgout, channels="GRAY", use_column_width=True, caption="Output Image", width=100)
            st.write("Kích thước ảnh: ", imgout.shape)
    else:
        st.warning("Vui lòng tải ảnh lên")

def xu_li_anh_c5():
    st.header("Khôi phục ảnh")
    imgin = None
    imgout = None
    file_uploaded = st.file_uploader("Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'tif'])
    if file_uploaded is not None:
        file_bytes = np.asarray(bytearray(file_uploaded.read()), dtype=np.uint8)
        imgin = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(imgin, channels="GRAY", use_column_width=True, caption="Input Image", width=100)
        st.write("Kích thước ảnh: ", imgin.shape)
    operation = st.selectbox("Các tùy chọn", ["", "CreateMotionNoise", "DenoiseMotion", "DenoisestMotion"])
    if imgin is not None:
        transfer = st.button("Chuyển đổi")
        if transfer:
            if operation == "CreateMotionNoise":
                imgout = CreateMotionNoise(imgin)
            elif operation == "DenoiseMotion":
                imgout = DenoiseMotion(imgin)
            elif operation == "DenoisestMotion":
                temp = cv2.medianBlur(imgin, 7)
                imgout = DenoiseMotion(temp)
            st.info("Ảnh sau khi xử lí")
            st.image(imgout, channels="GRAY", use_column_width=True, caption="Output Image", width=100)
            st.write("Kích thước ảnh: ", imgout.shape)
    else:
        st.warning("Vui lòng tải ảnh lên")

def xu_li_anh_c9():
    st.header("Xử lý ảnh hình thái")
    imgin = None
    imgout = None
    file_uploaded = st.file_uploader("Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'tif'])
    if file_uploaded is not None:
        file_bytes = np.asarray(bytearray(file_uploaded.read()), dtype=np.uint8)
        imgin = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(imgin, channels="GRAY", use_column_width=True, caption="Input Image", width=100)
        st.write("Kích thước ảnh: ", imgin.shape)
    operation = st.selectbox("Các tùy chọn", ["", "Connected Component", "Count Rice"])
    if imgin is not None:
        transfer = st.button("Chuyển đổi")
        if transfer:
            if operation == "Connected Component":
                text = ""
                imgout, text = ConnectedComponent(imgin, text)
                st.info(text)
            elif operation == "Count Rice":
                text = ""
                imgout, text = CountRice(imgin, text)
                st.info(text)
            st.info("Ảnh sau khi xử lí")
            st.image(imgout, channels="GRAY", use_column_width=True, caption="Output Image", width=100)
            st.write("Kích thước ảnh: ", imgout.shape)
    else:
        st.warning("Vui lòng tải ảnh lên")
