#%% importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

train = pd.read_csv('/Users/marcoistasy/Documents/Coding/imagine-ocr/object_detection/workspace/letters/annotations/train_labels.csv')
train.head()

image = plt.imread('/Users/marcoistasy/Documents/Coding/imagine-ocr/object_detection/workspace/letters/images/Image1.png')
plt.imshow(image)
plt.show()

# Number of unique training images
train['filename'].nunique()
train['class_type'].value_counts()

#%%
fig = plt.figure()

# add axes to the image
ax = fig.add_axes([0, 0, 1, 1])

# read and plot the image
image = plt.imread('/Users/marcoistasy/Documents/Coding/imagine-ocr/object_detection/workspace/letters/images/Image1.png')
plt.imshow(image)

# iterating over the image for different objects
for _, row in train[train.filename == "Image1.png"].iterrows():
    xmin = row.xmin
    xmax = row.xmax
    ymin = row.ymin
    ymax = row.ymax

    width = xmax - xmin
    height = ymax - ymin

    # assign different color to different classes of objects
    if row.class_type == 'o':
        edgecolor = 'r'
        ax.annotate('RBC', xy=(xmax - 40, ymin + 20))

    # add bounding boxes to the image
    rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')

    ax.add_patch(rect)

plt.show()

#%%

train = pd.read_csv('/Users/marcoistasy/Documents/Coding/imagine-ocr/object_detection/workspace/letters/keras-frcnn/train_labels.csv')
data = pd.DataFrame()
data['format'] = train['filename']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = '/Users/marcoistasy/Documents/Coding/imagine-ocr/object_detection/workspace/letters/keras' \
                        '-frcnn/train_images' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' \
                                                                                                        '' + str(
        train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['class_type'][i]

data.to_csv('/Users/marcoistasy/Documents/Coding/imagine-ocr/object_detection/workspace/letters/keras-frcnn/annotate.txt', header=None, index=None, sep=' ')
