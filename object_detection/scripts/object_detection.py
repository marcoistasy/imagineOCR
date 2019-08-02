#%%

# --------- Imports ---------

import numpy as np
import os
import tensorflow as tf
import cv2 as cv

from distutils.version import StrictVersion
from matplotlib import pyplot as plt
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

#%%

# --------- MODEL PREPARATION ---------

# todo: change model variables

PATH_TO_FROZEN_GRAPH = '/Users/marcoistasy/Documents/Coding/imagine-ocr/object_detection/workspace/letters_tf_object_detection/models/trained/ssd_inception_v2_coco/frozen_inference_graph.pb'
PATH_TO_LABELS = '/Users/marcoistasy/Documents/Coding/imagine-ocr/object_detection/workspace/letters_tf_object_detection/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)  # Label
# maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.

# load tf model

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

#%%

# --------- DETECTION ---------

# todo change detection variables

PATH_TO_TEST_IMAGES_DIR = '/Users/marcoistasy/Documents/Coding/imagine-ocr/object_detection/test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2)]
IMAGE_SIZE = (12, 8)  # Size, in inches, of the output images.


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:

            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}

            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


for image_path in TEST_IMAGE_PATHS:
    image = cv.imread(image_path)
    image_np_expanded = np.expand_dims(image, axis=0)  # Expand dimensions since the model expects images to have
    # shape: [1, None, None, 3]

    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

    # Visualisation of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np_expanded,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)

    # show the visualisation
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image)
    plt.show()
