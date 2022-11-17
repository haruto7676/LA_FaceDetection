import random

import cv2
import numpy as np

from util_func import overlayImage

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27  # Escキー
    SPACE_KEY = 32  # spaceキー
    INTERVAL = 10  # 待ち時間
    FRAME_RATE = 15  # fps

    # ウィンドウの命名
    WINDOW_NAME = "face_detect"
    cv2.namedWindow(WINDOW_NAME)

    # カメラ映像取得
    DEVICE_ID = 0
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # 分類器の指定
    cascade_file = "haarcascade_frontalface_alt2.xml"  # 目を基準に判定している分類器
    cascade = cv2.CascadeClassifier(cascade_file)

    # 被せる画像の指定（画像引用元 https://www.ac-illust.com/main/detail.php?id=1704191）
    cv2_img = []  # 配列にすべての画像を格納する
    for i in range(7):
        img = "img/fortune_0" + str(i + 1) + ".png"
        cv2_img.append(cv2.imread(img, cv2.IMREAD_UNCHANGED))

    count = 0  # 変換処理ループカウントの初期化
    count_boundary = 10  # 顔面非認識回数の境界

    drum_roll_counter = 0  # 画像がシャッフルされ続けるカウンター

    # 変換処理ループ
    while end_flag:
        face_list = cascade.detectMultiScale(c_frame, minSize=(100, 100))  # 顔を検出

        # 検出した顔に印を付ける
        if len(face_list) != 0 and drum_roll_counter < 20:
            face_list = np.sort(face_list, axis=0)
            for face_num, (x, y, w, h) in enumerate(face_list):  # face_numで何番目の顔か数えている
                # 重ね合わせたフレームを表示する
                c_frame = overlayImage(c_frame, cv2_img[random.randint(1, 6)], (x, y - h), (w, h))
                cv2.imshow(WINDOW_NAME, c_frame)
                drum_roll_counter = drum_roll_counter + 1

        elif len(face_list) != 0 and drum_roll_counter >= 20:
            # face_listをxの昇順で並び替え
            face_list = np.sort(face_list, axis=0)

            for face_num, (x, y, w, h) in enumerate(face_list):  # face_numで何番目の顔か数えている
                # 重ね合わせたフレームを表示する
                c_frame = overlayImage(c_frame, cv2_img[face_num], (x, y - h), (w, h))
                cv2.imshow(WINDOW_NAME, c_frame)

        else:
            # 顔が連続で検出されない回数
            count += 1
            # その回数がcount_boundaryを超えた場合
            if count >= count_boundary:
                cv2.imshow(WINDOW_NAME, c_frame)
                random.shuffle(cv2_img)  # 重ねる画像の変更

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        if cv2.waitKey(INTERVAL) == SPACE_KEY:
            drum_roll_counter = 0  # 初期化

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()

