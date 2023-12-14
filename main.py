import numpy as np
from PIL import Image


def image_process():
    cat_path = 'cat.jpg'
    cat = Image.open(cat_path)
    cat_numpy = np.array(cat)
    height, width, _ = cat_numpy.shape
    data = np.zeros(shape=(height * width, 3))
    threshold = 20
    for i in range(height):
        # start = 0
        # while start < width and np.sum(np.abs(cat_numpy[i][start])) < threshold:
        #     start += 1
        # end = width - 1
        # while end > start and np.sum(np.abs(cat_numpy[i][end])) < threshold:
        #     end -= 1
        # for j in range(0, start):
        #     data[i * width + j][0] = i
        #     data[i * width + j][1] = j
        # for j in range(start, end + 1):
        #     data[i * width + j][0] = i
        #     data[i * width + j][1] = j
        #     data[i * width + j][2] = 1
        # for j in range(end + 1, width):
        #     data[i * width + j][0] = i
        #     data[i * width + j][1] = j
        for j in range(width):
            data[i * width + j][0] = i
            data[i * width + j][1] = j
            if np.sum(np.abs(cat_numpy[i][j])) > threshold:
                data[i * width + j][2] = 1
    for i in range(height):
        for j in range(1, width - 1):
            if data[i * width + j - 1][2] and data[i * width + j + 1][2]:
                data[i * width + j][2] = 1
    data_grayscale = data[:, 2]
    data_grayscale = data_grayscale.reshape(height, width) * 255
    image = Image.fromarray(data_grayscale.astype(np.uint8))
    image.save('./cat/cat_grayscale.png')
    return data, data_grayscale


miku, miku_grayscale = image_process()


def save_image(data, height, width, filename):
    data = data.reshape((height, width)) * 255
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
    save_image(predict, miku_grayscale.shape[0], miku_grayscale.shape[1], './cat/tree_depth_{}.png'.format(depth))
    print('Finish tree_depth_{}'.format(depth))

from sklearn.ensemble import RandomForestClassifier

for depth in [5, 10, 15, 20]:
    rnd_clf = RandomForestClassifier(n_estimators=100, max_depth=depth, n_jobs=-1)  # n_jobs=-1指定cpu核数，可以并行训练
    rnd_clf.fit(miku_data, miku_target)
    predict = rnd_clf.predict(miku_data)
    save_image(predict, miku_grayscale.shape[0], miku_grayscale.shape[1], './cat/forest_depth_{}.png'.format(depth))
    print('Finish forest_depth_{}'.format(depth))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

for n in [10, 20, 50, 100]:
    forest_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=n)
    forest_clf.fit(miku_data, miku_target)
    forest_predict = forest_clf.predict(miku_data)
    save_image(forest_predict, miku_grayscale.shape[0], miku_grayscale.shape[1],
               './cat/adaboost_depth_5_{}.png'.format(n))
    print('Finish adaboost_depth_5_{}'.format(n))
