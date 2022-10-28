import cv2
from util_func import  overlayImage

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    WINDOW_NAME = "face_detect_ver1"

    DEVICE_ID = 0

    # 分類器の指定
    cascade_file = "haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(WINDOW_NAME)

    # 被せる画像の指定
    img = "nikoniko.png"
    cv2_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    

    # 変換処理ループ
    while end_flag == True:
        # 画像の取得と顔の検出
        face_list = cascade.detectMultiScale(c_frame, minSize=(100, 100))

          # 検出した顔に印を付ける
        if len(face_list) == 0:
            cv2.imshow('Frame', c_frame)
        else:
            for (x, y, w, h) in face_list:
                color = (0, 0, 225)
                pen_w = 3
                # cv2.rectangle(c_frame, (x, y), (x+w, y+h), color, thickness = pen_w)

                # 重ね合わせたフレームを表示する
                c_frame = overlayImage(c_frame, cv2_img, (x, y), (w, h))
                cv2.imshow('Frame', c_frame)

       

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()



    # 終了処理
    cv2.destroyAllWindows()
    cap.release()