import numpy as np
import pandas as pd
from PIL import Image

# 1st column: x, 2nd column: y, 3rd column: output (1 or 0)
miku = pd.read_csv('miku.csv', sep=' ', header=None)

miku = np.array(miku.values)

miku_grayscale = miku[:, 2]  # 得到类别
miku_grayscale = miku_grayscale.reshape((500, 500))  # 总共250000份数据
miku_grayscale = miku_grayscale.transpose()  # 这里进行装置，不然图像是倒着的

miku_grayscale = miku_grayscale * 255
image = Image.fromarray(miku_grayscale.astype(np.uint8))  # *255得到灰度值
image.save('./miku/miku_grayscale.png')


def save_image(data, height, width, filename):
    data = data.reshape((height, width)) * 255
    data = data.transpose()
    image_show = Image.fromarray(data.astype(np.uint8))
    image_show.save(filename)


# 拆分输入和输出
miku_data = miku[:, 0:2]  # 输入：坐标
miku_target = miku[:, -1]  # 输出：黑或白

from sklearn.tree import DecisionTreeClassifier

for depth in [5, 10, 15, 20]:
    dt_clf = DecisionTreeClassifier(max_depth=depth)
    dt_clf.fit(miku_data, miku_target)
    predict = dt_clf.predict(miku_data)
    save_image(predict, 500, 500, './miku/tree_depth_{}.png'.format(depth))
    print('Finish tree_depth_{}'.format(depth))

from sklearn.ensemble import RandomForestClassifier

for depth in [5, 10, 15, 20]:
    rnd_clf = RandomForestClassifier(n_estimators=100, max_depth=depth, n_jobs=-1)  # n_jobs=-1指定cpu核数，可以并行训练
    rnd_clf.fit(miku_data, miku_target)
    predict = rnd_clf.predict(miku_data)
    save_image(predict, 500, 500, './miku/forest_depth_{}.png'.format(depth))
    print('Finish forest_depth_{}'.format(depth))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

for n in [10, 20, 50, 100]:
    forest_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=n)
    forest_clf.fit(miku_data, miku_target)
    forest_predict = forest_clf.predict(miku_data)
    save_image(forest_predict, 500, 500,
               './miku/adaboost_depth_5_{}.png'.format(n))
    print('Finish adaboost_depth_5_{}'.format(n))
