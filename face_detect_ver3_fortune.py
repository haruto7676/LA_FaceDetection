import cv2
from util_func import  overlayImage
import random

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
    #cv2.namedWindow(WINDOW_NAME)

    # 被せる画像の指定
    # 画像引用元 https://www.ac-illust.com/main/detail.php?id=1704191
    
    cv2_img = [] # 配列にすべての画像を格納する
    for i in range(6):
        img = "img/fortune_0" + str(i+1) + ".png"
        cv2_img.append(cv2.imread(img, cv2.IMREAD_UNCHANGED))
    
    
    # 変換処理ループカウントの初期化
    count=0

    # 顔面非認識回数の境界
    count_boundary=10

    # 変換処理ループ
    while end_flag == True:
        # 画像の取得と顔の検出
        face_list = cascade.detectMultiScale(c_frame, minSize=(100, 100))

          # 検出した顔に印を付ける
        if len(face_list) != 0:
            count=0
            for face_num, (x, y, w, h) in enumerate(face_list): # face_numで何番目の顔か数えている
                color = (0, 0, 225)


                # 重ね合わせたフレームを表示する
                c_frame = overlayImage(c_frame, cv2_img[face_num], (x, y-h), (w, h))
                cv2.imshow('Frame', c_frame)

        else:
            #顔が連続で検出されない回数
            count=count+1

            #その回数がcount_boundaryを超えた場合
            if count>=count_boundary:
                cv2.imshow('Frame', c_frame)
                #重ねる画像を変えちゃうぞ？(画像が入ってる配列の要素の順番をシャッフル)
                random.shuffle(cv2_img)
                
            

       

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()



    # 終了処理
    cv2.destroyAllWindows()
    cap.release()