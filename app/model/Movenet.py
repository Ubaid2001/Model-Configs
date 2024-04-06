# Ubaid Mahmood
# S1906881

# Import Libraries
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# This class configures the MoveNet model, required for the pipeline.
class Movenet():

    image = None
    global interpreter

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

    # This function crops image with the input x and y points.
    # Return cropped_image - the processed image.
    #
    def crop_image(self, image, x1, y1, x2, y2, torso=False):
        image = Image.fromarray(image)
        cropped_image = None

        if x1 <= x2 and y1 <= y2:
            if torso is True:
                cropped_img = image.crop((x1, y1, x2, y2))
                print(cropped_img.size)
                numpy_image = np.array(cropped_img)

                cropped_image = numpy_image
            else:
                cropped_img = image.crop((x1 - 10, y1, x2 + 10, y2))
                numpy_image = np.array(cropped_img)

                cropped_image = numpy_image
        elif (x1 > x2) and (y1 > y2):
            if torso is True:
                cropped_img = image.crop((x2, y2, x1, y1))
                print(cropped_img.size)
                numpy_image = np.array(cropped_img)

                cropped_image = numpy_image
            else:
                cropped_img = image.crop((x2 - 10, y2, x1 + 10, y1))
                numpy_image = np.array(cropped_img)

                cropped_image = numpy_image
        elif (x1 > x2) and (y1 <= y2):
            if torso is True:
                cropped_img = image.crop((x2, y1, x1, y2))
                print(cropped_img.size)
                numpy_image = np.array(cropped_img)

                cropped_image = numpy_image
            else:
                cropped_img = image.crop((x2 - 10, y1, x1 + 10, y2))
                numpy_image = np.array(cropped_img)

                cropped_image = numpy_image
        elif (x1 <= x2) and (y1 > y2):

            if torso is True:
                cropped_img = image.crop((x1, y2, x2, y1))
                print(cropped_img.size)
                numpy_image = np.array(cropped_img)

                cropped_image = numpy_image
            else:
                cropped_img = image.crop((x1 - 10, y2, x2 + 10, y1))
                numpy_image = np.array(cropped_img)

                cropped_image = numpy_image

        return cropped_image

    # This function draws a circle on the keypoint of the person.
    #
    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        # y, x, c = frame.shape
        # shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        # This loop will identify all the pixels (keypoints) and print them if it exceeds the confidence threshold.
        # use to be for each in shaped
        for kp in keypoints:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

    # This function draws a line connnecting the keypoints together.
    # This function also crops the images into two sections, upperbody and lowerbody.
    # Return image1, image2 - upperbody image and lowerbody image.
    #
    def draw_connections(self, frame, keypoints, edges, confidence_threshold, original_image):
        # y, x, c = frame.shape
        # shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        left_shoulderX = None
        right_shoulderX = None
        left_hipY = None

        nose = None

        right_wristX = None
        left_wristX = None
        left_wristY = None

        right_elbowX = None
        left_elbowX = None

        left_ankleX = None
        left_ankleY = None
        right_ankleX = None

        image1 = None
        image2 = None


        # This loop will identify all the pixels (edge) if it exceeds the confidence threshold.
        # The keypoints var used to be shaped.
        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = keypoints[p1]
            y2, x2, c2 = keypoints[p2]

            if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # find the points and save it to a variable to be able to crop torso and bottoms.
                if p1 == 6 and p2 == 12:
                    print(f'edges: ({p1}, {p2})')
                    print("identified the right shoulder -> right hip bone structure")
                    right_shoulderX = x1

                elif p1 == 5 and p2 == 11:
                    print(f'edges: ({p1}, {p2})')
                    print("identified the left shoulder -> left hip bone structure")
                    left_shoulderX = x1
                    left_hipY = y2

                elif p1 == 0 and p2 == 1:
                    print(f'edges: ({p1}, {p2})')
                    print(f'identified the nose of the human')
                    nose = y1

                elif p1 == 8 and p2 == 10:
                    print(f'edges: ({p1}, {p2})')
                    print(f'identified the right elbow -> right wrist bone structure')
                    right_wristX = x2
                    right_elbowX = x1

                elif p1 == 7 and p2 == 9:
                    print(f'edges: ({p1}, {p2})')
                    print(f'identified the left elbow -> left wrist bone structure')
                    left_wristX = x2
                    left_wristY = y2
                    left_elbowX = x1

                elif p1 == 13 and p2 == 15:
                    print(f'edges: ({p1}, {p2})')
                    print(f'identified the left knee -> left ankle bone structure')
                    left_ankleX = x2
                    left_ankleY = y2

                elif p1 == 14 and p2 == 16:
                    print(f'edges: ({p1}, {p2})')
                    print(f'identified the right knee -> right ankle bone structure')
                    right_ankleX = x2

            else:
                print(f'Edge does not pass Movenet\'s confidence threshold ({p1}, {p2})')

        if left_wristX and left_hipY and right_wristX and left_elbowX and right_elbowX and nose is not None:

            if (left_wristX >= left_elbowX) and (right_wristX <= right_elbowX):
                print(f'self.crop_image(original_image, left_wristX, left_hipY, right_wristX, nose)')
                print(f'left_wristX: {left_wristX}, left_elbowX: {left_elbowX}, right_wristX: {right_wristX},'
                      f'right_elbowX: {right_elbowX} \n'
                      f'')
                image1 = self.crop_image(original_image, left_wristX, left_hipY, right_wristX, nose)
            elif (left_wristX < left_elbowX) and (right_wristX <= right_elbowX):
                print(f'self.crop_image(original_image, left_elbowX, left_hipY, right_wristX, nose)')
                print(f'left_wristX: {left_wristY}, left_elbowX: {left_elbowX}, right_wristX: {right_wristX},'
                      f'right_elbowX: {right_elbowX} \n'
                      f'')
                image1 = self.crop_image(original_image, left_elbowX, left_hipY, right_wristX, nose)
            elif (left_wristX >= left_elbowX) and (right_wristX > right_elbowX):
                print(f'self.crop_image(original_image, left_wristX, left_hipY, right_elbowX, nose)')
                print(f'left_wristX: {left_wristY}, left_elbowX: {left_elbowX}, right_wristX: {right_wristX},'
                      f'right_elbowX: {right_elbowX} \n'
                      f'')
                image1 = self.crop_image(original_image, left_wristX, left_hipY, right_elbowX, nose)
            else:
                print(f'self.crop_image(original_image, left_elbowX, left_hipY, right_elbowX, nose)')
                print(f'left_wristX: {left_wristY}, left_elbowX: {left_elbowX}, right_wristX: {right_wristX},'
                      f'right_elbowX: {right_elbowX} \n'
                      f'')
                image1 = self.crop_image(original_image, left_elbowX, left_hipY, right_elbowX, nose)
        elif left_shoulderX and left_hipY and right_shoulderX and nose is not None:

            print(f'self.crop_image(original_image, left_wristX, left_hipY, right_wristX, nose)')
            print(f'left_shoulderX: {left_shoulderX}, left_hipY: {left_hipY}, right_shoulderX: {right_shoulderX},'
                  f'nose: {nose} \n'
                  f'')
            image1 = self.crop_image(original_image, left_shoulderX, left_hipY, right_shoulderX, nose)


        if left_wristX and left_hipY and right_wristX and left_elbowX and right_elbowX and left_ankleY is not None:
            lfy = left_hipY * .15
            if (left_wristX >= left_elbowX) and (right_wristX <= right_elbowX):

                print(f'left_wristX: {left_wristX}, left_elbowX: {left_elbowX}, right_wristX: {right_wristX}, '
                      f'right_elbowX: {right_elbowX} \n'
                      f'')
                image2 = self.crop_image(original_image, left_wristX, left_hipY - lfy, right_wristX, left_ankleY)
            elif (left_wristX < left_elbowX) and (right_wristX <= right_elbowX):


                print(f'left_wristX: {left_wristY}, left_elbowX: {left_elbowX}, right_wristX: {right_wristX}, '
                      f'right_elbowX: {right_elbowX} \n'
                      f'')
                image2 = self.crop_image(original_image, left_elbowX, left_hipY - lfy, right_wristX, left_ankleY)
            elif (left_wristX >= left_elbowX) and (right_wristX > right_elbowX):


                print(f'left_wristX: {left_wristY}, left_elbowX: {left_elbowX}, right_wristX: {right_wristX}, '
                      f'right_elbowX: {right_elbowX} \n'
                      f'')
                image2 = self.crop_image(original_image, left_wristX, left_hipY - lfy, right_elbowX, left_ankleY)
            else:


                print(f'left_wristX: {left_wristY}, left_elbowX: {left_elbowX}, right_wristX: {right_wristX},'
                      f' right_elbowX: {right_elbowX} \n'
                      f'')
                image2 = self.crop_image(original_image, left_elbowX, left_hipY - lfy, right_elbowX, left_ankleY)

        elif left_hipY and right_elbowX and left_elbowX and right_shoulderX and left_ankleY \
                and left_shoulderX is not None:

            lfy = left_hipY * .15
            if (left_elbowX >= left_shoulderX) and (right_elbowX <= right_shoulderX):

                print(f'left_elbowX: {left_elbowX}, left_shoulderX: {left_shoulderX}, \n'
                      f'right_elbowX: {right_elbowX},'
                      f'right_shoulderX: {right_shoulderX} \n'
                      f'')


                image2 = self.crop_image(original_image, left_elbowX, left_hipY - lfy, right_elbowX, left_ankleY)
            elif (left_elbowX < left_shoulderX) and (right_elbowX <= right_shoulderX):

                print(f'left_elbowX: {left_elbowX}, left_shoulderX: {left_shoulderX}, \n'
                      f'right_elbowX: {right_elbowX},'
                      f'right_shoulderX: {right_shoulderX} \n'
                      f'')

                image2 = self.crop_image(original_image, left_shoulderX, left_hipY - lfy, right_elbowX, left_ankleY)
            elif (left_elbowX >= left_shoulderX) and (right_elbowX > right_shoulderX):

                print(f'left_elbowX: {left_elbowX}, left_shoulderX: {left_shoulderX}, \n'
                      f'right_elbowX: {right_elbowX},'
                      f'right_shoulderX: {right_shoulderX} \n'
                      f'')

                image2 = self.crop_image(original_image, left_elbowX, left_hipY - lfy, right_shoulderX, left_ankleY)
            else:

                print(f'left_elbowX: {left_elbowX}, left_shoulderX: {left_shoulderX}, \n'
                      f'right_elbowX: {right_elbowX},'
                      f'right_shoulderX: {right_shoulderX} \n'
                      f'')

                image2 = self.crop_image(original_image, left_shoulderX, left_hipY - lfy, right_shoulderX, left_ankleY)
        elif left_hipY and left_ankleY and left_ankleX and right_ankleX and left_shoulderX \
                and right_shoulderX is not None:

            print(f'right_ankleX: {right_ankleX}, left_ankleX: {left_ankleX}, \n'
                  f'left_hipY: {left_hipY},'
                  f'left_ankleY: {left_ankleY} \n'
                  f'')

            lfy = left_hipY * .15

            if (left_ankleX >= left_shoulderX) and (right_ankleX <= right_shoulderX):

                image2 = self.crop_image(original_image, left_ankleX, left_hipY - lfy, right_ankleX, left_ankleY)
            elif (left_ankleX < left_shoulderX) and (right_ankleX <= right_shoulderX):

                image2 = self.crop_image(original_image, left_shoulderX, left_hipY - lfy, right_ankleX, left_ankleY)
            elif (left_ankleX >= left_shoulderX) and (right_ankleX > right_shoulderX):

                image2 = self.crop_image(original_image, left_ankleX, left_hipY - lfy, right_shoulderX, left_ankleY)
            else:

                image2 = self.crop_image(original_image, left_shoulderX, left_hipY - lfy, right_shoulderX, left_ankleY)

        return image1, image2



    def get_affine_transform_to_fixed_sizes_with_padding(self, size, new_sizes):
        width, height = new_sizes
        scale = min(height / float(size[1]), width / float(size[0]))
        M = np.float32([[scale, 0, 0], [0, scale, 0]])
        M[0][2] = (width - scale * size[0]) / 2
        M[1][2] = (height - scale * size[1]) / 2
        return M

    # This function processes the input image.
    # Return image1, image2 - upperbody image and lowerbody image.
    #
    def get_image(self, frame):

        interpreter = tf.lite.Interpreter(model_path=os.path.abspath('lite-model_movenet_singlepose_lightning_3.tflite'))
        interpreter.allocate_tensors()

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
        image1, image2 = self.draw_connections(frame, keypoints_with_scores, self.EDGES, .2, original_image)
        self.draw_keypoints(frame, keypoints_with_scores, .2)

        return image1, image2
