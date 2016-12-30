import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import lxml.etree as ET
from skimage import transform as tf
import Config.char_alphabet as char_alpha
import random
from scipy import ndimage


def squeeze(image, width_max, border):
    """
    This function squeezes images in the width.
    :param image: Numpy array of image
    :param width_max: max image width
    :param border: border left and right of squeezed image
    :return: Squeezed Image
    """
    image_wd = image.shape[1]
    image_ht = image.shape[0]
    basewidth = width_max - border

    wpercent = basewidth / image_wd
    dim = (int(wpercent*image_wd), image_ht)
    img_squeeze= cv.resize(image, dim, interpolation=cv.INTER_NEAREST)
    return img_squeeze


def pad_label_with_blank(label, max_length):
    """
    This function takes string labels in form of integers and prepares them for use with Keras. Label will be converted 3
    to float and filled up with -1
    :param label: Label in form of integers. Example: label = [[51 38 48 45 43 45 50 43]] (obliging)
    :param max_length: Maximum length of string
    :return: label: label in form of [51. 38. 48. 45. 43. 45. 50. 43. -1. -1. -1.]
    :return: label_len_1: lenth of label
    """
    label_len_1 = len(label[0])
    label_len_2 = len(label[0])

    label_pad = []
    # label_pad.append(blank_id)
    for _ in label[0]:
        label_pad.append(_)
        # label_pad.append(blank_id)

    while label_len_2 < max_length:
        label_pad.append(-1)
        label_len_2 += 1

    label_out = np.ones(shape=[max_length]) #* np.asarray(blank_id)

    trunc = label_pad[:max_length]
    label_out[:len(trunc)] = trunc

    return label_out, label_len_1


def pad_sequence_into_array(image, width_max, border):
    """
    This function prepares image for CNN. It takes image and fits it to image size, e.g. image_ht x width_max.
    Randomly moves part of handwritten text to somewhere in the picture considering a border at the edge.
    :param image: image with text
    :param width_max: image width after padding
    :param border: border outside where part with text will not be moved to
    :return: Returns the image padded with 0
    """
    value = 0.
    image_ht = image.shape[0]
    image_wd = image.shape[1]
    offset_max = width_max - image_wd - border
    random_offset = random.randint(0, offset_max)

    Xout = np.ones(shape=[image_ht, width_max], dtype=image[0].dtype) * np.asarray(value, dtype=image[0].dtype)
    trunc = image[:, :width_max]
    Xout[:, random_offset:(random_offset+trunc.shape[1])] = trunc

    return Xout


def random_noise(img):  #TODO dataug
    """
    Puts random noise on image
    :param img: image without noise
    :return: image with noise
    """
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0

    return img_speck


def string_to_array(label_string):
    """
    This function converts string into integers. e.g. [hallo] --> [8,1,12,12,15]
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
    This function takes an image as input and displays it.
    :param img: an image import via opencv2
    """
    plt.imshow(img)
    plt.show()


def XML_load_word(filepath, filename):
    """
    This function is used for loading labels out of corresponding xml file
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
    :param tupel_filenames: [(path to img_file, path to xml, filename)]
    :param is_line: different loading for lines within IAM dataset
    :return: returns image and label
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
    This function detects skew in images. It turns the image so that the baseline of image is straight.
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

    # Calculate angle by looking at gradient of linear function + data augmentation
    angle = np.arctan(a) * 180 / np.pi + random.uniform(-1, 1) #TODO dataug

    # Rotate image and use Nearest Neighbour for interpolation of pixel
    rows, cols = img.shape
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv.warpAffine(img, M, (cols, rows), flags=cv.INTER_NEAREST)

    return img_rot


def slant(img):
    """
    Creates random slant on images for data augmentation
    :param img: image
    :return: slanted image
    """
    # Create random slant for data augmentation
    slant_factor = random.uniform(-0.2, 0.2)

    # Create Afine transform
    afine_tf = tf.AffineTransform(shear=slant_factor)

    # Apply transform to image data
    img_slanted = tf.warp(img, afine_tf, order=0)
    return img_slanted


def pad_border(img, add_pix_wd, add_pix_ht):
    """
    Pads image so that text is not directly on edge
    :param img: image
    :param add_pix_wd: # of pixels on top and bottom
    :param add_pix_ht: # of pixels left and right
    :return: image with border
    """
    value = 255.

    image_ht = img.shape[0]
    img_white_space1 = np.ones(shape=[image_ht, add_pix_wd], dtype=img[0].dtype) * np.asarray(value, dtype=img[0].dtype)

    img_out1 = np.concatenate((img_white_space1,img),axis=1)
    img_out2 = np.concatenate((img_out1, img_white_space1),axis=1)

    image_wd = img_out2.shape[1]
    img_white_space2 = np.ones(shape=[add_pix_ht, image_wd], dtype=img[0].dtype) * np.asarray(value, dtype=img[0].dtype)

    img_out3 = np.concatenate((img_white_space2, img_out2),axis=0)
    img_out4 = np.concatenate((img_out3, img_white_space2), axis=0)

    return img_out4


def scaling(img):
    """
    This function scales the image down so that height is exactly 64 pixel. The width of every image may vary.
    :param img: image
    :return: scaled image
    """
    baseheight = 64  #TODO config
    hpercent = (baseheight / float(img.shape[0]))
    dim = (int(img.shape[1] * hpercent), baseheight)

    img_scaled = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)

    return img_scaled


def prep_run(input_tuple, is_line):
    """ TODO:
    This function takes an image as Input. During Pre-Processing following steps are computed:
        1. Load image and label
        2. Greyscale
        3. Pad border
        4. Thresholding
        5. Skew
        6. Slant
        7. Scaling
        8. Squeezing
        9. pad_sequence_into_array
        10. random_noise
        11. string_to_array
        12. pad_label_with_blank
        13. Batch
    :param input_tuple: [(path to img_file, path to xml, filename)]
    :return output_tuple: [img_noise, label_blank, label_len, label]
    """
    batch = []
    # Input Parameters
    chars = char_alpha.chars
    #output_size = len(chars)

    for input in input_tuple:
        # print ("Inputs: ", input)
        # 1. Load img and label
        img_raw, label_raw = load(input, is_line)

        # Image prep
        # 2. Greyscale
        img_grey = greyscale(img_raw, input)
        # 3. Pad a border
        img_white = pad_border(img_grey, 15, 10)
        # 4. Thresholding
        img_thresh = thresholding(img_white)
        # 5. Skew
        img_skew = skew(img_thresh)
        # 6. Data augmentation slant
        img_slant = slant(img_skew)
        # 7. Scaling
        img_scal = scaling(img_slant)
        # 8. Squeeze
        if img_scal.shape[1] > 256 - 10:
            img_scal = squeeze(img_scal, 256, 10)
        # 9. Padding into full size
        img_pad = pad_sequence_into_array(img_scal, 256, 10) #TODO
        # # 10. Data augmentation random noise
        # img_noise = random_noise(img_pad)

        # Label prep
        # 11. Convert label string to array of int
        label = string_to_array(label_raw)
        # 12. Pad label with blank (-1)
        label_blank, label_len = pad_label_with_blank(label, 40) #TODO

        # Batch
        # 13. Include to batch
        batch.append([img_pad, label_blank, label_len, label])

    # print("Preprocessing successful! Batchsize: ", len(input_tuple))
    return batch
