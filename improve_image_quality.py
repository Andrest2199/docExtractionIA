import os
from PIL import Image
import numpy as np
import cv2
import math
from typing import Tuple, Union
from deskew import determine_skew
from skimage import io, filters

"""
    Additional documentation:
    https://www.leadtools.com/help/sdk/v21/main/api/deskewing.html
"""

debug = True


def get_skew_angle(cvImage) -> float:
    """ "
    Get the skew angle of the image using Hough Transform
    return the average angle of the lines in the image
    rtype: float
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=250
    )
    angles = []
    # calculate the angle of each line
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)

    # return the average angle
    if not angles:
        return None
    return np.mean(angles)


def deskew_and_rotate(image, output_path):
    """ "
    Deskew and rotate the image
    rtpe: PIL.Image.Image
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    rotated = rotate(image, angle, (0, 0, 0))
    cv2.imwrite(output_path, rotated)
    output_image = Image.open(output_path)
    return output_image


def convert_to_1bit(image):
    """ "
    Convert the image to 1-bit (black and white)
    rtype: PIL.Image.Image
    """
    np_image = np.array(image)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    # Use Sauvola's method to get better results
    sauvola = filters.threshold_sauvola(np_image, window_size=15)
    binary_image = np_image > sauvola
    return binary_image


def convert_to_grayscale(image):
    """
    Convert the image to grayscale
    rtype: PIL.Image.Image
    """
    np_image = np.array(image)
    grayscale_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def rotate(
    image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(
        np.cos(angle_radian) * old_width
    )
    height = abs(np.sin(angle_radian) * old_width) + abs(
        np.cos(angle_radian) * old_height
    )

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(
        image, rot_mat, (int(round(height)), int(round(width))), borderValue=background
    )


def get_bounding_boxes(contours, mask, textImg):
    """
    Auxiliar function to
    Get the bounding boxes of the text regions
    """
    # Initialize min and max coordinates
    min_x, min_y, max_x, max_y = float("inf"), float("inf"), 0, 0
    cummTheta = 0
    ct = 0
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y : y + h, x : x + w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y : y + h, x : x + w])) / (w * h)

        if r > 0.45 and w > 8 and h > 8:
            rect = cv2.minAreaRect(contours[idx])
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            # cv2.drawContours(textImg, [box], 0, (0, 0, 255), 2)

            theta = slope(box[0][0], box[0][1], box[1][0], box[1][1])
            cummTheta += theta
            ct += 1

            # Update min and max coordinates
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

    # Check for float("inf") after the loop
    if min_x == float("inf"):
        min_x = 0
    if min_y == float("inf"):
        min_y = 0

    return int(min_x), int(min_y), int(max_x), int(max_y), cummTheta, ct


def slope(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    slope = (y2 - y1) / (x2 - x1)
    theta = np.rad2deg(np.arctan(slope))
    return theta


def display(img, frameName="OpenCV Image"):
    if not debug:
        return
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww * (h / w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
    cv2.waitKey(0)


def improve_image_quality(input_image_path, output_image_path):
    filename = os.path.basename(input_image_path)
    output_image_path = os.path.join(output_image_path, filename)
    raw_image = cv2.imread(input_image_path)

    if raw_image is None:
        print(f"Image {input_image_path} not found")
        return None

    # crop image to text region
    small = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

    # find the gradient map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    # Binarize the gradient image
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # connect horizontally oriented regions
    # kernal value (9,1) can be changed to improved the text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )  # opencv >= 4.0
    mask = np.zeros(bw.shape, dtype=np.uint8)
    min_x, min_y, max_x, max_y, cummTheta, ct = get_bounding_boxes(
        contours, mask, raw_image
    )
    cropped = raw_image[min_y:max_y, min_x:max_x]
    # display(cropped, "Cropped Image")

    skewed_angle = get_skew_angle(cropped)
    # display(cropped, "Cropped Image")
    deskewed_image = cropped  # assign a default value
    if skewed_angle is not None:
        if abs(skewed_angle) > 0.5:
            print(f"Image is skewed by {skewed_angle} degrees")
            deskewed_image = deskew_and_rotate(cropped, output_image_path)
    else:
        deskewed_image = cropped
    # bin_img = convert_to_1bit(deskewed_image)
    bin_img = convert_to_grayscale(deskewed_image)
    # bin_img = (bin_img * 255).astype(np.uint8)
    cv2.imwrite(output_image_path, bin_img)
