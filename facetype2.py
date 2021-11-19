import os
import cv2
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np

classes = ["直線", "曲線"]
classes2 = ["大人", "子供"]
num_classes = len(classes)
num_classes = len(classes2)
image_size = 64


# UPLOAD_FOLDER = "uploads"
UPLOAD_FOLDER = "C:\\Users\\na-09\\2022\\webapp2022\\opencv_work\\facetype\\testpic" #uploadされた写真を保存するところ
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__, static_folder='static')

#ファイルの拡張子が正しいかどうかの確認
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#.があるかのチェックと拡張子の確認　OKなら１，だめなら0


# 顔を検出して顔部分の画像（64x64）を返す関数
def detect_face(img):
    # 画像をグレースケールへ変換
    img_gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # カスケードファイルのパス
    cascade_path = "C:\\Users\\na-09\\2022\\webapp2022\\opencv_work\\haarcascades\\haarcascades\\haarcascade_frontalface_default.xml"
    # カスケード分類器の特徴量取得
    cascade = cv2.CascadeClassifier(cascade_path)
    # 顔認識
    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(10,10))
    
    # 顔認識出来なかった場合
    if len(faces) == 0:
        face = faces
    # 顔認識出来た場合
    else:
        # 顔部分画像を取得
        for x,y,w,h in faces:
            face = img[y:y+h, x:x+w]
        # リサイズ
        face = cv2.resize(face, (image_size, image_size))
    return face


# 学習済みモデルをロードする
model = load_model("C:\\Users\\na-09\\2022\\webapp2022\\opencv_work\\facetype\\model_wavee.h5")
model_2 = load_model("C:\\Users\\na-09\\2022\\webapp2022\\opencv_work\\facetype\\model_child.h5")

graph = tf.get_default_graph()

@app.route('/')
def index():
 return render_template("toppage.html")

#ファイルを受け取る方法の指定
@app.route('/index.html', methods=['GET', 'POST'])
def upload_file():
        if request.method == 'POST': #リクエストがポストかどうかの判別
            if 'file' not in request.files: #ファイルがなかったとき
                flash('ファイルがありません')
                return redirect(request.url)
            file = request.files['file'] #データの取り出し
            if file.filename == '': #ファイル名がなかったとき
                flash('ファイルがありません')
                return redirect(request.url)
            if file and allowed_file(file.filename):#ファイルのチェック
                filename = secure_filename(file.filename) #危険な文字を削除（サニタイズ処理
                file.save(os.path.join(UPLOAD_FOLDER, filename)) #ファイルを保存
                filepath = os.path.join(UPLOAD_FOLDER, filename)
            


                img = cv2.imread(filepath, 1)  #受け取った画像を読み込
                img = detect_face(img)  # 顔検出して大きさ64x64

                # 顔認識出来なかった場合
                if len(img) == 0:
                    pred_answer = "顔を検出できませんでした。他の画像を送信してください。"
                    return render_template("index.html",answer=pred_answer) #第一引数にHTML名を指定
                # 顔認識出来た場合
                else:
                    # 画像の保存
                    image_path = UPLOAD_FOLDER + "/face_" + file.filename
                    cv2.imwrite(image_path, img)

                    img = image.img_to_array(img) #配列に変換
                    data = np.array([img])
                    
                    result = model.predict(data)[0] #予測 #cool&cute
                    print(result)
                    predicted = result.argmax()
                    result = model_2.predict(data)[0] #予測 #flesh&feminine
                    print(result)
                    predicted_2 = result.argmax()
                    pred_answer = classes[predicted] +"&" + classes2[predicted_2] + "顔タイプです"
                    message_comment = "顔を検出できていない場合は他の画像を送信してください"

                    return render_template("index.html", answer = pred_answer,  message = message_comment)
                #画像をWEBページに表示する


        return render_template("index.html",answer="")


if __name__ == "__main__":
    app.debug = True
    app.run()
