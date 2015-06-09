import cv2
import numpy
import math

from numpy.random import random_integers
from scipy.signal import convolve2d

def prepare_test_image(image, width ,resize_shape, negated=False):
    """
    This function normalizes an an already padded image and flattens it into a
    row vector
    :param image: the input image
    :type image: numpy nd array

    :param resize_shape: a tuple denoting the shape of the padded image
    :type resize_shape: tuple

    :param negated: a flag to know if the input image is a negated one
    :type negated: boolean

    :returns : a 1-D array
    """

    # negate the image 
    if not negated:
        image = 255-image
    
    # resizing the image 
    resized_image = resize_img(image, resize_shape, negated=True)
    #resized_image = width_normalization(image, width, resize_shape, negated=True)
    
    # gaussian filtering
    resized_image = cv2.GaussianBlur(resized_image,(3,3), 0)
    # deskew
    #deskewed_image = deskew(resized_image, resize_shape)
    
    # normalize the image values to fit in the range [0,1]
    norm_image = numpy.asarray(resized_image, dtype=numpy.float32) / 255.

    # Flatten the image to a 1-D vector and return
    return norm_image.reshape(1, resize_shape[0] * resize_shape[1])



def do_cropping(image, negated=False):
    """
    This method will crop the image using the outermost detectable contour
    
    :param image: input image
    :type image: numpy array

    :param negated: a boolean value indicating whether the image is already
                    negated one
    :type negated: boolean
    """
    
    # if the image has 3 channels, convert it into a single channel one
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check if the image is already negated. If not negate it
    if not negated:
        image = 255-image

    # do thresholding
    ret,thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    # find contours
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)

    # find the index of contour with maximum area
    try:
        max_index = numpy.argmax(numpy.asarray([len(c) for c in contours]))
    except ValueError:
        return image

    # find the cropping co-ordinates
    x, y, width, height = cv2.boundingRect(contours[max_index])

    # return cropped image
    cropped_img = image[y:y+height, x:x+width]
    
    # the cropped image should of the same format as input image
    if not negated:
        cropped_img = 255-cropped_img

    return cropped_img


def deskew(image, image_shape, negated=False):
    """
    This method deskwes an image using moments
    :param image: a numpy nd array input image
    :param image_shape: a tuple denoting the image`s shape
    :param negated: a boolean flag telling  whether the input image is a negated one

    :returns: a numpy nd array deskewd image
    """
    
    # negate the image
    if not negated:
        image = 255-image

    # calculate the moments of the image
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()

    # caclulating the skew
    skew = m['mu11']/m['mu02']
    M = numpy.float32([[1, skew, -0.5*image_shape[0]*skew], [0,1,0]])
    img = cv2.warpAffine(image, M, image_shape, flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    
    return img


def resize_img(image, target_shape, value=255, min_padding=2, negated=False):
    """
    This method adds padding to the image and makes it to a nxn array,
    without losing the aspect ratio
    
    :param image: the input image
    :type image: numpy array

    :param target_shape: the dimensions to which the image needs to be resized
    :type target_shape: tuple

    :param min_padding: minimum padding that to be added
    :type min_padding: int

    :param value: the value of the padding area, 0-black, 255-white
    :type value: int
    
    :param negated: a flag indicating the input image is a negated one or not
    :type negated: bool

    :returns :  a padded image 
    """
    
    # if the image is a multi channel one, convert it into a single channel one
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # If the input image is already neagted, then the padding should be black
    if negated:
        value = 0

    # image dimensions
    image_height, image_width = image.shape

    # target dimensions
    target_height, target_width = target_shape 

    # Add padding
    # The aim is to make an image of different width and height to a sqaure image
    # For that first the biggest attribute among width and height are determined.
    max_index = numpy.argmax([image_height, image_width])

    # if height is the biggest one, then add padding to width until width becomes
    # equal to height
    if max_index == 0:
        padded_img = cv2.copyMakeBorder(image, min_padding, min_padding,
                                        (image_height + min_padding - image_width)/2, 
                                        (image_height + min_padding - image_width)/2, 
                                        cv2.BORDER_CONSTANT, value=value)
    
    # else if width is the biggest one, then add padding to height until height becomes
    # equal to width
    else:
        padded_img = cv2.copyMakeBorder(image, 
                                        (image_width + min_padding - image_height)/2, 
                                        (image_width + min_padding - image_height)/2, 
                                        min_padding, min_padding, cv2.BORDER_CONSTANT, 
                                        value=value)
    
    # finally resize the sqaure image to the target shape
    return cv2.resize(padded_img, target_shape)


def create_2d_gaussian(dim, sigma):
    """
    This function creates a 2d gaussian kernel with the standard deviation
    denoted by sigma
    
    :param dim: integer denoting a side (1-d) of gaussian kernel
    :type dim: int

    :param sigma: the standard deviation of the gaussian kernel
    :type sigma: float
    
    :returns: a numpy 2d array
    """

    # check if the dimension is odd
    if dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # initialize the kernel
    kernel = numpy.zeros((dim, dim), dtype=numpy.float16)

    # calculate the center point
    center = dim/2

    # calculate the variance
    variance = sigma ** 2
    
    # calculate the normalization coefficeint
    coeff = 1. / (2 * variance)

    # create the kernel
    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val**2 + y_val**2
            denom = 2*variance
            
            kernel[x,y] = coeff * numpy.exp(-1. * numerator/denom)
    
    # normalise it
    return kernel/sum(sum(kernel))


def elastic_transform(image, kernel_dim=13, sigma=6, alpha=36, negated=False):
    """
    This method performs elastic transformations on an image by convolving 
    with a gaussian kernel.

    NOTE: Image dimensions should be a sqaure image
    
    :param image: the input image
    :type image: a numpy nd array

    :param kernel_dim: dimension(1-D) of the gaussian kernel
    :type kernel_dim: int

    :param sigma: standard deviation of the kernel
    :type sigma: float

    :param alpha: a multiplicative factor for image after convolution
    :type alpha: float

    :param negated: a flag indicating whether the image is negated or not
    :type negated: boolean

    :returns: a nd array transformed image
    """
    
    # convert the image to single channel if it is multi channel one
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check if the image is a negated one
    if not negated:
        image = 255-image

    # check if the image is a square one
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image should be of sqaure form")

    # check if kernel dimesnion is odd
    if kernel_dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # create an empty image
    result = numpy.zeros(image.shape)

    # create random displacement fields
    displacement_field_x = numpy.array([[random_integers(-1, 1) for x in xrange(image.shape[0])] \
                            for y in xrange(image.shape[1])]) * alpha
    displacement_field_y = numpy.array([[random_integers(-1, 1) for x in xrange(image.shape[0])] \
                            for y in xrange(image.shape[1])]) * alpha

    # create the gaussian kernel
    kernel = create_2d_gaussian(kernel_dim, sigma)

    # convolve the fields with the gaussian kernel
    displacement_field_x = convolve2d(displacement_field_x, kernel)
    displacement_field_y = convolve2d(displacement_field_y, kernel)

    # make the distortrd image by averaging each pixel value to the neighbouring
    # four pixels based on displacement fields
    
    for row in xrange(image.shape[1]):
        for col in xrange(image.shape[0]):
            low_ii = row + int(math.floor(displacement_field_x[row, col]))
            high_ii = row + int(math.ceil(displacement_field_x[row, col]))

            low_jj = col + int(math.floor(displacement_field_y[row, col]))
            high_jj = col + int(math.ceil(displacement_field_y[row, col]))

            if low_ii < 0 or low_jj < 0 or high_ii >= image.shape[1] -1 \
               or high_jj >= image.shape[0] - 1:
                continue

            res = image[low_ii, low_jj]/4 + image[low_ii, high_jj]/4 + \
                    image[high_ii, low_jj]/4 + image[high_ii, high_jj]/4

            result[row, col] = res
    
    # if the input image was not negated, make the output image also a non 
    # negated one
    if not negated:
        result = 255-result

    return result

        
def width_normalization(image, width, target_shape, negated=False):
    """
    This method creates a width normalised 1-d vector of an image
    
    :param image: the input image
    :type image: numpy nd array

    :param width: the width to which the image should be normalized 
                  (a value of -1 will just crop the image along its contour)
    :type width: int 

    :param target_shape: a tuple denoting the output dims
    :type target_shape: tuple

    :returns: a nd array width normalized image
    """
    
    # if the image have 3 channels, then convert it into grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # negate the image
    if not negated:
        image = 255-image

    # crop the number bounding box
    cropped_img = do_cropping(image, negated=True)

    if not (cropped_img.shape[0] * cropped_img.shape[1]):
        cropped_img = image

    # width normalization
    if width == -1:
        width_normalized_img = cropped_img
    else:
        width_normalized_img = cv2.resize(cropped_img, (width, cropped_img.shape[1]))
    
    # add padding and resize to the specified shape
    resized_image = resize_img(width_normalized_img, target_shape, negated=True)

    # return the width normalized image
    if not negated:
        resized_image = 255-resized_image 
    
    return resized_image

