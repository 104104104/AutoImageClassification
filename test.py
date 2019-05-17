import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import glob, os
from sklearn.cluster import KMeans
from PIL import Image

#vgg16の構築
model=VGG16()
model.summary()


# 画像を読み込み、モデルの入力サイズでリサイズする。
path_li=glob.glob("img/*")
predss=[]
print(path_li)
for img_path in path_li:

    img = image.load_img(img_path, target_size=model.input_shape[1:3])

    # PIL.Image オブジェクトを np.float32 型の numpy 配列に変換する。
    x = image.img_to_array(img)

    # 配列の形状を (Height, Width, Channels) から (1, Height, Width, Channels) に変更する。
    x = np.expand_dims(x, axis=0)

    # VGG16 用の前処理を行う。
    x = preprocess_input(x)

    predss.append(model.predict(x)[0])

np_predss=np.array(predss)

pred = KMeans(n_clusters=6).fit_predict(predss)

print(pred)

for i in range(len(path_li)):
    imgpath=path_li[i]
    num=str(pred[i])
    if not os.path.exists(num):
        os.makedirs(num+"/img/")
        imgfile=Image.open(imgpath, "r")
        imgfile.save(num+"/"+imgpath)
    else:
       imgfile=Image.open(imgpath, "r")
       imgfile.save(num+"/"+imgpath)
