import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import cv2 as cv
import lxml.etree as ET  #import xml.etree.cElementTree as ET
from skimage import transform as tf
import Config.char_alphabet as char_alpha


def label_preproc(label_string):
    """
    This function is supposed to prepare the label so that it fits the standard of the rnn_ctc network.
    It computes following steps:
    1. make list of integers out of string    e.g. [hallo] --> [8,1,12,12,15]
    :param label_string: a string of the label
    :return: label_int: the string represented in integers
    """
    chars = char_alpha.chars

    label_int = []

    for letter in label_string:
        label_int.append(chars.index(letter))

    label_int_arr = np.resize(np.asarray(label_int), (1, len(label_int)))

    # print(label_int_arr.shape)

    return label_int_arr

def show_img(img):
    """
    This function takes an image as input and displays it. It is used for testing stages of preprocessing
    :param img: an image import via opencv2
    """
    plt.imshow(img)
    plt.show()


def XML_load_word(filepath, filename):
    """
    This funtion is used for loading labels out of corresponding xml file
    :param filepath: location of xml fidle
    :param filename: the name of the current image
    :return: the label of the image
    """
    with open(filepath, 'rb') as xml_file:
        tree = ET.parse(xml_file)
        # tree = ET.parse(filepath)
        root = tree.getroot()

    # source = open(filepath)
    # tree = ET.parse(source)
    # root = tree.getroot()

    for child in root.iter('word'):
        if child.get('id') == filename:
            return child.get('text')
    # source.close()


def XML_load_line(filepath, filename):
    """
    This funtion is used for loading labels out of corresponding xml file
    :param filepath: location of xml file
    :param filename: the name of the current image
    :return: the label of the image
    """
    with open(filepath, 'rb') as xml_file:
        tree = ET.parse(xml_file)
        # tree = ET.parse(filepath)
        root = tree.getroot()

    # source = open(filepath)
    # tree = ET.parse(source)
    # root = tree.getroot()

    for child in root.findall('./handwritten-part/'):
        if child.get('id') == filename:
            return child.get('text')
    # source.close()


def load(tupel_filenames, is_line):
    """
    Load image and label
    :param tupel_filenames:
    :param is_line:
    :return:
    """

    # returns None when filepath is false
    img = cv.imread(tupel_filenames[0])
    if is_line == 0:
        label = XML_load_word(tupel_filenames[1], tupel_filenames[2])
    else:
        label = XML_load_line(tupel_filenames[1], tupel_filenames[2])
    return img, label


def greyscale(img, input):
    """
    Makes a greyscale image out of a normal image
    :param img: colored image
    :return: img_grey: greyscale image
    """
    try:
        img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    except:
        print("CV-Error bei Input: ", input)
        # r06-022-03-05
        # a01-117-05-02
    return img_grey


def thresholding(img_grey):
    """
    This functions creates binary images using thresholding
    :param img_grey: greyscale image
    :return: binary image
    """
    # # Adaptive Gaussian
    # img_binary = cv.adaptiveThreshold(img_grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img_grey, (5, 5), 0)
    ret3, img_binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # invert black = 255
    ret, thresh1 = cv.threshold(img_binary, 127, 255, cv.THRESH_BINARY_INV)

    return thresh1


def skew(img):
    """
    This function detects skew in images. It turn the image so that the baseline of image is straight.
    :param img: the image
    :return: rotated image
    """
    # coordinates of bottom black pixel in every column
    black_pix = np.zeros((2, 1))

    # Look at image column wise and in every column from bottom to top pixel. It stores the location of the first black
    # pixel in every column
    for columns in range(img.shape[1]):
        for pixel in np.arange(img.shape[0]-1, -1, -1):
            if img[pixel][columns] == 255:
                black_pix = np.concatenate((black_pix, np.array([[pixel], [columns]])), axis=1)
                break

    # Calculate linear regression to detect baseline
    mean_x = np.mean(black_pix[1][:])
    mean_y = np.mean(black_pix[0][:])
    k = black_pix.shape[1]
    a = (np.sum(black_pix[1][:] * black_pix[0][:]) - k * mean_x * mean_y) / (np.sum(black_pix[1][:] * black_pix[1][:]) - k * mean_x * mean_x)

    # Calculate angle by looking at gradient of linear function
    angle = np.arctan(a) * 180 / np.pi

    # Rotate image and use Nearest Neighbour for interpolation of pixel
    rows, cols = img.shape
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv.warpAffine(img, M, (cols, rows), flags=cv.INTER_NEAREST)

    return img_rot


def slant(img):
    # Load the image as a matrix
    """

    :param img:
    :return:
    """
    # Create Afine transform
    afine_tf = tf.AffineTransform(shear=0.1)  #TODO which factor???

    # Apply transform to image data
    img_slanted = tf.warp(img, afine_tf, order=0)
    return img_slanted


def positioning(img):
    """

    :param img:
    :return:
    """
    return img


def scaling(img):
    """
    This function scale the image down so that height is exactly 40 pixel. Th width of every image may vary.
    :param img:
    :return: resized image
    """
    baseheight = 64  # data_config.img_ht  #TODO is data always being created?
    hpercent = (baseheight / float(img.shape[0]))
    dim = (int(img.shape[1] * hpercent), baseheight)

    img_scaled = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)

    # print(img_scaled.shape)

    return img_scaled


def prep_run(input_tuple, is_line):
    """ TODO:
    This function takes an image as Input. During Pre-Processing following steps are computed:
        1. Load image and label
        2. Greyscale
        3. Thresholding
        4. Skew
        5. Slant
        6. Positioning
        7. Scaling
        8. Preprocessing of label
    :param input_tuple: [path to img_file, path to xml]
    :return output_tuple: [normalized image of text line, label]
    """
    batch = []

    # print("Batchsize: ", len(input_tuple))



    for input in input_tuple:
        # print ("Inputs: ", input)
        # 1. Load img and label
        img_raw, label_raw = load(input, is_line)
        # 2. Greyscale
        img_grey = greyscale(img_raw, input)
        # 3. Thresholding
        img_thresh = thresholding(img_grey)
        # 4. Skew
        img_skew = skew(img_thresh)
        # 5. Slant
        img_slant = slant(img_skew)
        # 6. Positioning
        img_pos = positioning(img_slant)
        # 7. Scaling
        img_norm = scaling(img_pos)
        # 8. Preprocessing of label
        label = label_preproc(label_raw)
        # 9. Include to batch
        batch.append([img_norm, label, label_raw])

    # print("Preprocessing successful! Batchsize: ", len(input_tuple))
    return batch
