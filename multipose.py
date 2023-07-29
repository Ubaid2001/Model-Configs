# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# import matplotlib.patches as patches
# import math
# import cv2
# import random
# import os
# import imageio
# from IPython.display import HTML, display
#
# interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_multipose_lightning_tflite_float16_1.tflite')
# interpreter.allocate_tensors()
#
# print(interpreter.get_signature_runner('serving_default'))
#
# model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
# movenet = model.signatures['serving_default']
#
# # model = hub.load('./lite-model_movenet_multipose_lightning_tflite_float16_1.tflite')
# # movenet = model.signatures['serving_default']
#
#
# # def detect(interpreter, input_tensor):
# #   """Runs detection on an input image.
# #
# #   Args:
# #     interpreter: tf.lite.Interpreter
# #     input_tensor: A [1, input_height, input_width, 3] Tensor of type tf.float32.
# #       input_size is specified when converting the model to TFLite.
# #
# #   Returns:
# #     A tensor of shape [1, 6, 56].
# #   """
# #
# #   input_details = interpreter.get_input_details()
# #   output_details = interpreter.get_output_details()
# #
# #   is_dynamic_shape_model = input_details[0]['shape_signature'][2] == -1
# #   if is_dynamic_shape_model:
# #     input_tensor_index = input_details[0]['index']
# #     input_shape = input_tensor.shape
# #     interpreter.resize_tensor_input(
# #         input_tensor_index, input_shape, strict=True)
# #   interpreter.allocate_tensors()
# #
# #   interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())
# #
# #   interpreter.invoke()
# #
# #   keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
# #   return keypoints_with_scores
#
# # def keep_aspect_ratio_resizer(image, target_size):
# #   """Resizes the image.
# #
# #   The function resizes the image such that its longer side matches the required
# #   target_size while keeping the image aspect ratio. Note that the resizes image
# #   is padded such that both height and width are a multiple of 32, which is
# #   required by the model.
# #   """
# #   _, height, width, _ = image.shape
# #   if height > width:
# #     scale = float(target_size / height)
# #     target_height = target_size
# #     scaled_width = math.ceil(width * scale)
# #     image = tf.image.resize(image, [target_height, scaled_width])
# #     target_width = int(math.ceil(scaled_width / 32) * 32)
# #   else:
# #     scale = float(target_size / width)
# #     target_width = target_size
# #     scaled_height = math.ceil(height * scale)
# #     image = tf.image.resize(image, [scaled_height, target_width])
# #     target_height = int(math.ceil(scaled_height / 32) * 32)
# #   image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)
# #   return (image,  (target_height, target_width))
#
# def draw_keypoints(frame, keypoints, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
#
#     for kp in shaped:
#         ky, kx, kp_conf = kp
#         if kp_conf > confidence_threshold:
#             cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)
#
# def draw_connections(frame, keypoints, edges, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
#
#     for edge, color in edges.items():
#         p1, p2 = edge
#         y1, x1, c1 = shaped[p1]
#         y2, x2, c2 = shaped[p2]
#
#         if (c1 > confidence_threshold) & (c2 > confidence_threshold):
#             cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
#
# # Function to loop through each person detected and render
# def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
#     for person in keypoints_with_scores:
#         draw_connections(frame, person, edges, confidence_threshold)
#         draw_keypoints(frame, person, confidence_threshold)
#
# # Dictionary that maps from joint names to keypoint indices.
# KEYPOINT_DICT = {
#     'nose': 0,
#     'left_eye': 1,
#     'right_eye': 2,
#     'left_ear': 3,
#     'right_ear': 4,
#     'left_shoulder': 5,
#     'right_shoulder': 6,
#     'left_elbow': 7,
#     'right_elbow': 8,
#     'left_wrist': 9,
#     'right_wrist': 10,
#     'left_hip': 11,
#     'right_hip': 12,
#     'left_knee': 13,
#     'right_knee': 14,
#     'left_ankle': 15,
#     'right_ankle': 16
# }
#
# EDGES = {
#     (0, 1): 'm',
#     (0, 2): 'c',
#     (1, 3): 'm',
#     (2, 4): 'c',
#     (0, 5): 'm',
#     (0, 6): 'c',
#     (5, 7): 'm',
#     (7, 9): 'm',
#     (6, 8): 'c',
#     (8, 10): 'c',
#     (5, 6): 'y',
#     (5, 11): 'm',
#     (6, 12): 'c',
#     (11, 12): 'y',
#     (11, 13): 'm',
#     (13, 15): 'm',
#     (12, 14): 'c',
#     (14, 16): 'c'
# }
#
# # cap = cv2.VideoCapture('novak.mp4')
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     # Resize image
#     img = frame.copy()
#     img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384, 640)
#     input_img = tf.cast(img, dtype=tf.int32)
#     # input_img = tf.cast(img, dtype=tf.uint8)
#     # print(input_img.shape)
#
#     # # Setup input and output
#     # input_details = interpreter.get_input_details()
#     # output_details = interpreter.get_output_details()
#
#     # Detection section
#     results = movenet(input_img)
#     keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
#
#     # # Make predictions
#     # interpreter.set_tensor(input_details[0]['index'], input_img)
#     # interpreter.invoke()
#     # keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
#
#     # # Resize and pad the image to keep the aspect ratio and fit the expected size.
#     # resized_image, image_shape = keep_aspect_ratio_resizer(input_img, 256)
#     # image_tensor = tf.cast(resized_image, dtype=tf.uint8)
#     #
#     # # Output: [1, 6, 56] tensor that contains keypoints/bbox/scores.
#     # keypoints_with_scores = detect(
#     #     interpreter, tf.cast(image_tensor, dtype=tf.uint8))
#
#     # Render keypoints
#     # loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
#     loop_through_people(frame, keypoints_with_scores, EDGES, 0.4)
#
#     cv2.imshow('Movenet Multipose', frame)
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
#
# # # Load the input image.
# # input_size = 256
# # il = os.listdir("imdb-images/04")
# # il = np.array(il)
# # # image = cv2.imread("./imdb-images/04/" + il[random.choice(range(0, len(il)))])
# # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # image = tf.io.read_file("./imdb-images/04/" + il[random.choice(range(0, len(il)))])
# # image = tf.compat.v1.image.decode_jpeg(image)
# # image = tf.expand_dims(image, axis=0)
# # # image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.float32)
# #
# #
# # # Resize and pad the image to keep the aspect ratio and fit the expected size.
# # resized_image, image_shape = keep_aspect_ratio_resizer(image, input_size)
# # image_tensor = tf.cast(resized_image, dtype=tf.uint8)
# #
# # # Output: [1, 6, 56] tensor that contains keypoints/bbox/scores.
# # keypoints_with_scores = detect(
# #     interpreter, image_tensor)
# #
# # print(keypoints_with_scores.shape)
# # print(keypoints_with_scores[0])
# #
# # # Visualize the predictions with image.
# # display_image = image
# # display_image = tf.cast(tf.image.resize_with_pad(
# #     display_image, 1280, 1280), dtype=tf.int32)
# # output_overlay = draw_prediction_on_image(
# #     np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)
# #
# # plt.figure(figsize=(5, 5))
# # plt.imshow(output_overlay)
# # _ = plt.axis('off')
# # plt.show()


import tensorflow as tf
import numpy as np
import cv2

interpreter = tf.lite.Interpreter(
    model_path="lite-model_movenet_singlepose_lightning_3.tflite"
)
interpreter.allocate_tensors()


def draw_keypoints(frame, keypoints, confidence_threshold):
    # y, x, c = frame.shape
    # shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    # print(f'This is the shaped data {shaped}')
    # print(f'This is the shaped shape {shaped.shape}')

    # This loop will identify all the pixels (keypoints) and print them if it exceeds the confidence threshold.
    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
            print(f'This is the kx {kx}\n '
                  f'This is the ky {ky}\n'
                  f'This is the kp_conf {kp_conf}')


def get_affine_transform_to_fixed_sizes_with_padding(size, new_sizes):
    width, height = new_sizes
    scale = min(height / float(size[1]), width / float(size[0]))
    M = np.float32([[scale, 0, 0], [0, scale, 0]])
    M[0][2] = (width - scale * size[0]) / 2
    M[1][2] = (height - scale * size[1]) / 2
    return M


frame = cv2.imread("man_pics/man2.jpg")

# Reshape image
img = frame.copy()
img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
input_image = tf.cast(img, dtype=tf.float32)

# Setup input and output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Make predictions
interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
interpreter.invoke()
keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])[0, 0]

img_resized = np.array(input_image).astype(np.uint8)[0]
keypoints_for_resized = keypoints_with_scores.copy()
keypoints_for_resized[:, 0] *= img_resized.shape[1]
keypoints_for_resized[:, 1] *= img_resized.shape[0]
draw_keypoints(img_resized, keypoints_for_resized, 0.4)
cv2.imwrite("image_with_keypoints_resized.png", img_resized)

orig_w, orig_h = frame.shape[:2]
M = get_affine_transform_to_fixed_sizes_with_padding((orig_w, orig_h), (192, 192))
# M has shape 2x3 but we need square matrix when finding an inverse
M = np.vstack((M, [0, 0, 1]))
M_inv = np.linalg.inv(M)[:2]
xy_keypoints = keypoints_with_scores[:, :2] * 192
xy_keypoints = cv2.transform(np.array([xy_keypoints]), M_inv)[0]
keypoints_with_scores = np.hstack((xy_keypoints, keypoints_with_scores[:, 2:]))

# Rendering
draw_keypoints(frame, keypoints_with_scores, 0.4)
cv2.imwrite("image_with_keypoints_original.png", frame)