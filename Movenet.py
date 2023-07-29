import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import cv2
import math
import os
import random
from PIL import Image

class Movenet():

    image = None
    global interpreter
    # Hello Dude 1dfa
    # interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
    # interpreter.allocate_tensors()

    EDGES = {
        (0, 1): 'm',
        (0, 2): 'c',
        (1, 3): 'm',
        (2, 4): 'c',
        (0, 5): 'm',
        (0, 6): 'c',
        (5, 7): 'm',
        (7, 9): 'm',
        (6, 8): 'c',
        (8, 10): 'c',
        (5, 6): 'y',
        (5, 11): 'm',
        (6, 12): 'c',
        (11, 12): 'y',
        (11, 13): 'm',
        (13, 15): 'm',
        (12, 14): 'c',
        (14, 16): 'c'
    }

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

    # def __init__(self, image):
    #     self.image = image


    def crop_image(self, image, x1, y1, x2, y2, torso=False):
        image = Image.fromarray(image)
        width, height = image.size

        if x1 <= x2 and y1 <= y2:
            if torso is True:
                cropped_img = image.crop((x1, y1, x2, y2))
                print(cropped_img.size)
                numpy_image = np.array(cropped_img)

                plt.imshow(numpy_image)
                plt.show()
            else:
                cropped_img = image.crop((x1 - 10, y1, x2 + 10, y2))
                numpy_image = np.array(cropped_img)

                plt.imshow(numpy_image)
                plt.show()
        elif (x1 > x2) and (y1 > y2):
            if torso is True:
                cropped_img = image.crop((x2, y2, x1, y1))
                print(cropped_img.size)
                numpy_image = np.array(cropped_img)

                plt.imshow(numpy_image)
                plt.show()
            else:
                cropped_img = image.crop((x2 - 10, y2, x1 + 10, y1))
                numpy_image = np.array(cropped_img)

                plt.imshow(numpy_image)
                plt.show()
        elif (x1 > x2) and (y1 <= y2):
            if torso is True:
                cropped_img = image.crop((x2, y1, x1, y2))
                print(cropped_img.size)
                numpy_image = np.array(cropped_img)

                plt.imshow(numpy_image)
                plt.show()
            else:
                cropped_img = image.crop((x2 - 10, y1, x1 + 10, y2))
                numpy_image = np.array(cropped_img)

                plt.imshow(numpy_image)
                plt.show()
        elif (x1 <= x2) and (y1 > y2):

            if torso is True:
                cropped_img = image.crop((x1, y2, x2, y1))
                print(cropped_img.size)
                numpy_image = np.array(cropped_img)

                plt.imshow(numpy_image)
                plt.show()
            else:
                cropped_img = image.crop((x1 - 10, y2, x2 + 10, y1))
                numpy_image = np.array(cropped_img)

                plt.imshow(numpy_image)
                plt.show()

        # cropped_img = image.crop((w1, h1, width, height))
        #
        # image.show()
        # print(image)
        # print(cropped_img)
        # cropped_img.show()
        # numpy_image = np.array(cropped_img)
        # opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        # opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        #
        # plt.imshow(numpy_image)
        # plt.show()


    # def crop_image_torso(image, x1, y1, x2, y2, torso=False):
    #     image = Image.fromarray(image)
    #     width, height = image.size
    #
    #     if x1 <= x2 and y1 <= y2:
    #         if torso is True:
    #             cropped_img = image.crop((x1, y1, x2, y2))
    #             print(cropped_img.size)
    #             numpy_image = np.array(cropped_img)
    #
    #             plt.imshow(numpy_image)
    #             plt.show()
    #         else:
    #             cropped_img = image.crop((x1 - 10, y1, x2 + 10, y2))
    #             numpy_image = np.array(cropped_img)
    #
    #             plt.imshow(numpy_image)
    #             plt.show()
    #     elif (x1 > x2) and (y1 > y2):
    #         if torso is True:
    #             cropped_img = image.crop((x2, y2, x1, y1))
    #             print(cropped_img.size)
    #             numpy_image = np.array(cropped_img)
    #
    #             plt.imshow(numpy_image)
    #             plt.show()
    #         else:
    #             cropped_img = image.crop((x2 - 10, y2, x1 + 10, y1))
    #             numpy_image = np.array(cropped_img)
    #
    #             plt.imshow(numpy_image)
    #             plt.show()
    #     elif (x1 > x2) and (y1 <= y2):
    #         if torso is True:
    #             cropped_img = image.crop((x2, y1, x1, y2))
    #             print(cropped_img.size)
    #             numpy_image = np.array(cropped_img)
    #
    #             plt.imshow(numpy_image)
    #             plt.show()
    #         else:
    #             cropped_img = image.crop((x2 - 10, y1, x1 + 10, y2))
    #             numpy_image = np.array(cropped_img)
    #
    #             plt.imshow(numpy_image)
    #             plt.show()
    #     elif (x1 <= x2) and (y1 > y2):
    #
    #         if torso is True:
    #             cropped_img = image.crop((x1, y2, x2, y1))
    #             print(cropped_img.size)
    #             numpy_image = np.array(cropped_img)
    #
    #             plt.imshow(numpy_image)
    #             plt.show()
    #         else:
    #             cropped_img = image.crop((x1 - 10, y2, x2 + 10, y1))
    #             numpy_image = np.array(cropped_img)
    #
    #             plt.imshow(numpy_image)
    #             plt.show()
    #
    #     # cropped_img = image.crop((w1, h1, width, height))
    #     # print(image.size)
    #     # w1 = x1 - x2
    #     # h1 = y1 - y2
    #     # # cropped_img = image.crop((y2, x2, y1, x1))
    #     # cropped_img = image.crop((x2, y2, x1, y1))
    #     # print(cropped_img.size)
    #     # numpy_image = np.array(cropped_img)
    #
    #     # plt.imshow(numpy_image)
    #     # plt.show()






    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        # y, x, c = frame.shape
        # shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        # print(f'This is the shaped data {shaped}')
        # print(f'This is the shaped shape {shaped.shape}')

        # This loop will identify all the pixels (keypoints) and print them if it exceeds the confidence threshold.
        # use to be for each in shaped
        for kp in keypoints:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
                # print(f'This is the kx {kx}\n '
                #       f'This is the ky {ky}\n'
                #       f'This is the kp_conf {kp_conf}')
                # print(f'this is the keypoints {kp}')


    def draw_connections(self, frame, keypoints, edges, confidence_threshold, original_image):
        # y, x, c = frame.shape
        # shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        left_shoulder = None
        right_shoulder = None
        left_hip = None
        right_hip = None
        # global original_image

        # This loop will identify all the pixels (edge) if it exceeds the confidence threshold.
        # The keypoints var used to be shaped.
        for edge, color in edges.items():
            # print(f'this is the edge {edge}\n'
            #       f'this is the colour {color}')
            p1, p2 = edge
            y1, x1, c1 = keypoints[p1]
            y2, x2, c2 = keypoints[p2]

            if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                print(f'this is the edge {edge}\n'
                      f'this is the colour {color}')
                # print(f'this is the p1 {p1}\n'
                #       f'this is the p2 {p2}')
                print(f'this is the p1: x1 {x1} & y1: {y1} {p1}\n'
                      f'this is the p2: x2 {x2} & y2: {y2}  {p2}')

                match (p1, p2):
                    case (5, 7):
                        print("identified the left shoulder -> left elbow bone structure")
                        self.crop_image(original_image, x1, y1, x2, y2)
                        # left_shoulder = x1
                    case (7, 9):
                        print("identified the left elbow -> left wrist bone structure")
                        self.crop_image(original_image, x1, y1, x2, y2)
                    case (6, 8):
                        print("identified the right shoulder -> right elbow bone structure")
                        self.crop_image(original_image, x1, y1, x2, y2)
                        # right_shoulder = x1
                    case (8, 10):
                        print("identified the right elbow -> right wrist bone structure")
                        self.crop_image(original_image, x1, y1, x2, y2)
                    case (11, 13):
                        print("identified the left hip -> left knee bone structure")
                        self.crop_image(original_image, x1, y1, x2, y2)
                        # left_hip = x1
                    case (13, 15):
                        print("identified the left knee -> left ankle bone structure")
                        self.crop_image(original_image, x1, y1, x2, y2)
                    case (12, 14):
                        print("identified the right hip -> right knee bone structure")
                        self.crop_image(original_image, x1, y1, x2, y2)
                        # right_hip = x1
                    case (14,  16):
                        print("identified the right knee -> right ankle bone structure")
                        self.crop_image(original_image, x1, y1, x2, y2)
                    case (6, 12):
                        print("identified the right shoulder -> right hip bone structure")
                        # # crop_image(original_image, x1, y1, x2, y2)
                        # right_shoulder = x1
                        # right_hip = x2
                        right_shoulder = x1
                        # This is not the right hip x coordinate but is instead the right shoulder y coordinate
                        right_hip = y1
                    case (5, 11):
                        print("identified the left shoulder -> left hip bone structure")
                        # crop_image(original_image, x1, y1, x2, y2)
                        # left_shoulder = x1
                        # left_hip = y2
                        left_shoulder = x1
                        left_hip = y2

        # the slight inaccuracy of the cropping could be due to the addition of pixels in the crop_image method.
        if left_shoulder and left_hip and right_shoulder and right_hip is not None:
            print(f'identified the torso of the person')
            # print(left_hip, left_shoulder, right_hip, right_shoulder)
            print(left_shoulder, left_hip, right_shoulder, right_hip)
            # crop_image(original_image, left_hip, left_shoulder, right_hip, right_shoulder)
            self.crop_image(original_image, left_shoulder, left_hip, right_shoulder, right_hip, torso=True)
            # crop_image_torso(original_image, left_shoulder, left_hip, right_shoulder, right_hip, torso=True)

    def get_affine_transform_to_fixed_sizes_with_padding(self, size, new_sizes):
        width, height = new_sizes
        scale = min(height / float(size[1]), width / float(size[0]))
        M = np.float32([[scale, 0, 0], [0, scale, 0]])
        M[0][2] = (width - scale * size[0]) / 2
        M[1][2] = (height - scale * size[1]) / 2
        return M


    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #
    #     # Reshape image
    #     img = frame.copy()
    #     img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    #     input_image = tf.cast(img, dtype=tf.float32)
    #
    #     # Setup input and output
    #     input_details = interpreter.get_input_details()
    #     output_details = interpreter.get_output_details()
    #
    #     # Make predictions
    #     interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    #     interpreter.invoke()
    #     keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    #
    #     # Rendering
    #     draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    #     draw_keypoints(frame, keypoints_with_scores, 0.4)
    #
    #     cv2.imshow('MoveNet Lightning', frame)
    #
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()


    # # The threshold could be changed to .4 so that the guess should be more than 40% correct to display.
    # def _keypoints_and_edges_for_display(keypoints_with_scores,
    #                                      height,
    #                                      width,
    #                                      keypoint_threshold=0.11):
    #   """Returns high confidence keypoints and edges for visualization.
    #
    #   Args:
    #     keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
    #       the keypoint coordinates and scores returned from the MoveNet model.
    #     height: height of the image in pixels.
    #     width: width of the image in pixels.
    #     keypoint_threshold: minimum confidence score for a keypoint to be
    #       visualized.
    #
    #   Returns:
    #     A (keypoints_xy, edges_xy, edge_colors) containing:
    #       * the coordinates of all keypoints of all detected entities;
    #       * the coordinates of all skeleton edges of all detected entities;
    #       * the colors in which the edges should be plotted.
    #   """
    #   keypoints_all = []
    #   keypoint_edges_all = []
    #   edge_colors = []
    #   print(keypoints_with_scores.shape)
    #   num_instances, _, _, _ = keypoints_with_scores.shape
    #   for idx in range(num_instances):
    #     kpts_x = keypoints_with_scores[0, idx, :, 1]
    #     kpts_y = keypoints_with_scores[0, idx, :, 0]
    #     kpts_scores = keypoints_with_scores[0, idx, :, 2]
    #     kpts_absolute_xy = np.stack(
    #         [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    #     kpts_above_thresh_absolute = kpts_absolute_xy[
    #         kpts_scores > keypoint_threshold, :]
    #     keypoints_all.append(kpts_above_thresh_absolute)
    #
    #     for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
    #       if (kpts_scores[edge_pair[0]] > keypoint_threshold and
    #           kpts_scores[edge_pair[1]] > keypoint_threshold):
    #         x_start = kpts_absolute_xy[edge_pair[0], 0]
    #         y_start = kpts_absolute_xy[edge_pair[0], 1]
    #         x_end = kpts_absolute_xy[edge_pair[1], 0]
    #         y_end = kpts_absolute_xy[edge_pair[1], 1]
    #         line_seg = np.array([[x_start, y_start], [x_end, y_end]])
    #         keypoint_edges_all.append(line_seg)
    #         edge_colors.append(color)
    #   if keypoints_all:
    #     keypoints_xy = np.concatenate(keypoints_all, axis=0)
    #   else:
    #     keypoints_xy = np.zeros((0, 17, 2))
    #
    #   if keypoint_edges_all:
    #     edges_xy = np.stack(keypoint_edges_all, axis=0)
    #   else:
    #     edges_xy = np.zeros((0, 2, 2))
    #   return keypoints_xy, edges_xy, edge_colors

    # def draw_prediction_on_image(
    #     image, keypoints_with_scores, crop_region=None, close_figure=False,
    #     output_image_height=None):
    #   """Draws the keypoint predictions on image.
    #
    #   Args:
    #     image: A numpy array with shape [height, width, channel] representing the
    #       pixel values of the input image.
    #     keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
    #       the keypoint coordinates and scores returned from the MoveNet model.
    #     crop_region: A dictionary that defines the coordinates of the bounding box
    #       of the crop region in normalized coordinates (see the init_crop_region
    #       function below for more detail). If provided, this function will also
    #       draw the bounding box on the image.
    #     output_image_height: An integer indicating the height of the output image.
    #       Note that the image aspect ratio will be the same as the input image.
    #
    #   Returns:
    #     A numpy array with shape [out_height, out_width, channel] representing the
    #     image overlaid with keypoint predictions.
    #   """
    #   height, width, channel = image.shape
    #   aspect_ratio = float(width) / height
    #   fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    #   # To remove the huge white borders
    #   fig.tight_layout(pad=0)
    #   ax.margins(0)
    #   ax.set_yticklabels([])
    #   ax.set_xticklabels([])
    #   plt.axis('off')
    #
    #   im = ax.imshow(image)
    #   line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    #   ax.add_collection(line_segments)
    #   # Turn off tick labels
    #   scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)
    #
    #   (keypoint_locs, keypoint_edges,
    #    edge_colors) = _keypoints_and_edges_for_display(
    #        keypoints_with_scores, height, width, .4)
    #
    #   line_segments.set_segments(keypoint_edges)
    #   line_segments.set_color(edge_colors)
    #   if keypoint_edges.shape[0]:
    #     line_segments.set_segments(keypoint_edges)
    #     line_segments.set_color(edge_colors)
    #   if keypoint_locs.shape[0]:
    #     scat.set_offsets(keypoint_locs)
    #
    #   if crop_region is not None:
    #     xmin = max(crop_region['x_min'] * width, 0.0)
    #     ymin = max(crop_region['y_min'] * height, 0.0)
    #     rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    #     rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    #     rect = patches.Rectangle(
    #         (xmin,ymin),rec_width,rec_height,
    #         linewidth=1,edgecolor='b',facecolor='none')
    #     ax.add_patch(rect)
    #
    #   fig.canvas.draw()
    #   image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #   image_from_plot = image_from_plot.reshape(
    #       fig.canvas.get_width_height()[::-1] + (3,))
    #   plt.close(fig)
    #   if output_image_height is not None:
    #     output_image_width = int(output_image_height / height * width)
    #     image_from_plot = cv2.resize(
    #         image_from_plot, dsize=(output_image_width, output_image_height),
    #          interpolation=cv2.INTER_CUBIC)
    #   return image_from_plot


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


    # This obtains a random image and performs the pose estimation on a single person
    # ===============================================================================
    # il = os.listdir("imdb-images/04")
    # il = np.array(il)
    # # print(il[0])
    #
    # ran_img = cv2.imread("./imdb-images/04/" + il[random.choice(range(0, len(il)))])
    # ran_img = cv2.cvtColor(ran_img, cv2.COLOR_BGR2RGB)
    #
    #
    # # Resize and pad the image to keep the aspect ratio and fit the expected size.
    # input_image = tf.expand_dims(ran_img, axis=0)
    # input_image = tf.cast(tf.image.resize_with_pad(input_image, 192, 192), dtype=tf.float32)
    #
    # # Run model inference.
    #
    # # Setup input and output
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    #
    # # Make predictions
    # interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    # interpreter.invoke()
    # keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    #
    # # Visualize the predictions with image.
    # display_image = tf.expand_dims(ran_img, axis=0)
    # display_image = tf.cast(tf.image.resize_with_pad(
    #     display_image, 1280, 1280), dtype=tf.int32)
    # output_overlay = draw_prediction_on_image(
    #     np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)
    #
    # plt.figure(figsize=(5, 5))
    # plt.imshow(output_overlay)
    # _ = plt.axis('off')
    # plt.show()


    # This is the male prediction side of it
    # !!!!!!DO NOT USE | SAVING FOR LATER USE!!!!!!!
    # ==============================================
    # faces = extract_faces(ran_img)
    # # print(faces)
    #
    # predict_X = []
    # for face in faces:
    #     predict_X.append(face['face'])
    #
    # predict_X = np.array(predict_X)
    #
    # predictions = []
    # if predict_X.shape[0] > 0:
    #     predictions = model.predict(predict_X)
    #
    # show_output(ran_img, faces, predictions)


    #  This is for capturing the single pose via the camera
    # ======================================================

    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #
    #     # Reshape image
    #     img = frame.copy()
    #     img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    #     input_image = tf.cast(img, dtype=tf.float32)
    #
    #     # Setup input and output
    #     input_details = interpreter.get_input_details()
    #     output_details = interpreter.get_output_details()
    #
    #     # Make predictions
    #     interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    #     interpreter.invoke()
    #     keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    #
    #     # Rendering
    #     draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    #     draw_keypoints(frame, keypoints_with_scores, 0.4)
    #
    #     cv2.imshow('MoveNet Lightning', frame)
    #
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()


    def get_image(self, frame):

        interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
        interpreter.allocate_tensors()

        # frame = cv2.imread("man_pics/man2.jpg")
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # il = os.listdir("imdb-images/00/")
        # il = np.array(il)
        # frame = cv2.imread("./imdb-images/00/" + il[random.choice(range(0, len(il)))])
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # # Reshape image
        # img = frame.copy()
        # img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        # input_image = tf.cast(img, dtype=tf.float32)

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


        orig_w, orig_h = frame.shape[:2]
        M = self.get_affine_transform_to_fixed_sizes_with_padding((orig_w, orig_h), (192, 192))
        # M has shape 2x3 but we need square matrix when finding an inverse
        M = np.vstack((M, [0, 0, 1]))
        M_inv = np.linalg.inv(M)[:2]
        xy_keypoints = keypoints_with_scores[:, :2] * 192
        xy_keypoints = cv2.transform(np.array([xy_keypoints]), M_inv)[0]
        keypoints_with_scores = np.hstack((xy_keypoints, keypoints_with_scores[:, 2:]))

        # Rendering
        original_image = frame.copy()
        self.draw_connections(frame, keypoints_with_scores, self.EDGES, .4, original_image)
        self.draw_keypoints(frame, keypoints_with_scores, 0.4)
        plt.imshow(frame)
        plt.show()
