import cv2
import streamlit as st
import numpy as np
import joblib
import argparse

def str2bool(v):
        if v.lower() in ['on', 'yes', 'true', 'y', 't']:
            return True
        elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
            return False
        else:
            raise NotImplementedError

def nhan_dang_khuon_mat():

    # List of recognized names
    global mydict
    mydict = ['Bao Quoc', 'Thai Hung', 'Thanh Vinh', 'Minh Quan','Quang Truong']
    batdau= st.checkbox('Bắt đầu nhận dạng')
    FRAME_WINDOW = st.image([])
    global camera
    camera = cv2.VideoCapture(0)

    svc = joblib.load('./model/nhan_dang_khuon_mat/model/svc.pkl')
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
    parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
    parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
    parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
    parser.add_argument('--face_detection_model', '-fd', type=str, default='./model/nhan_dang_khuon_mat/model/face_detection_yunet_2023mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
    parser.add_argument('--face_recognition_model', '-fr', type=str, default='./model/nhan_dang_khuon_mat/model/face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
    parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
    parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
    parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
    parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
    global args
    args = parser.parse_args()

        
    def visualize(input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                cv2.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

                if faces[1] is not None:
                    # Resize the frame to [320 x 320] for face detection
                    face_align = recognizer.alignCrop(frame, face)  # Use the current face
                    face_feature = recognizer.feature(face_align)
                    test_predict = svc.predict(face_feature)
                    result = mydict[test_predict[0]]
                    # Draw the "result" text above each rectangle for each face
                    cv2.putText(input, result, (coords[0], coords[1] - 10 - idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    detector = cv2.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (720, 620),  # Set the input size to match the model's expectation
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )

    # Load face recognition model
    recognizer = cv2.FaceRecognizerSF.create(
        args.face_recognition_model,
        ""
    )


    tm = cv2.TickMeter()

    while batdau:
        _, frame = camera.read()
        frame = cv2.resize(frame, (720, 620))
        tm.start()
        faces = detector.detect(frame)  # faces is a tuple
        tm.stop()
        
        visualize(frame, faces, tm.getFPS())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')
