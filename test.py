import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image

#vgg16の構築
model=VGG16()
model.summary()

print('model.input_shape', model.input_shape)  # model.input_shape (None, 224, 224, 3)

# 画像を読み込み、モデルの入力サイズでリサイズする。
img_path = input()
img = image.load_img(img_path, target_size=model.input_shape[1:3])

# PIL.Image オブジェクトを np.float32 型の numpy 配列に変換する。
x = image.img_to_array(img)
print('x.shape: {}, x.dtype: {}'.format(x.shape, x.dtype))
# x.shape: (224, 224, 3), x.dtype: float32

# 配列の形状を (Height, Width, Channels) から (1, Height, Width, Channels) に変更する。
x = np.expand_dims(x, axis=0)
print('x.shape: {}'.format(x.shape))  # x.shape: (1, 224, 224, 3)

# VGG16 用の前処理を行う。
x = preprocess_input(x)



preds = model.predict(x)
print('preds.shape: {}'.format(preds.shape))  # preds.shape: (1, 1000)


result = decode_predictions(preds, top=3)[0]
print(result)

for _, name, score in result:
    print('{}: {:.2%}'.format(name, score))
