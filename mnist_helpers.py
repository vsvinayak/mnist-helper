import cv2
import numpy
import math

from numpy.random import random_integers
from scipy.signal import convolve2d

def prepare_test_image(image, width ,resize_shape, negated=False):
    """
    This function normalizes an an already padded image and flattens it into a
    row vector
    :param image: a numpy array
    :param resize_shape: a tuple denoting the shape of the padded image
    :param negated: a flag to know if the input image is a negated one
    :return : a 1-D array
    """

    # negate the image 
    if not negated:
        image = 255-image

    # crop the image
    cropped_image = do_cropping(image)

    # resizing the image 
    resized_image = resize_img(image, resize_shape, negated=True)
    #resized_image = width_normalization(image, width, resize_shape, negated=negated)
    
    # median filtering
    resized_image = cv2.GaussianBlur(resized_image,(3,3), 0)
    # deskew
    #deskewed_image = deskew(resized_image, resize_shape)
    
    # normalize the image values to fit in the range [0,1]
    norm_image = numpy.asarray(resized_image, dtype=numpy.float32) / 255.

    # Flatten the image to a 1-D vector and return
    return norm_image.reshape(1, resize_shape[0] * resize_shape[1])



def do_cropping(image):
    """
    This method will crop the image using the outermost detectable contour
    
    :param image: a numpy array
    """

    # do thresholding
    ret,thresh = cv2.threshold(255-image,127,255,0)

    # find contours
    img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,
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


def resize_img(image, target_shape, value=255, min_padding=5, negated=False):
    """
    This method adds padding to the image and makes it to a nxn array,
    without losing the aspect ratio
    
    :param image: a numpy array
    :param target_shape: the dimensions to which the image needs to be resized
    :param min_padding: minimum padding that to be added
    :param value: the value of the padding area
    :returns :  a padded image 
    """

    # set the value for negated
    if negated:
        value = 0

    # Getting the image dimensions
    image_height, image_width = image.shape

    # Getting the target dimensions
    target_height, target_width = target_shape 

    # Add padding
    max_index = numpy.argmax([image_height, image_width])
    if max_index == 0:
        padded_img = cv2.copyMakeBorder(image, min_padding, min_padding,
                                        (image_height + min_padding - image_width)/2, 
                                        (image_height + min_padding - image_width)/2, 
                                        cv2.BORDER_CONSTANT, value=value)
    else:
        padded_img = cv2.copyMakeBorder(image, 
                                        (image_width + min_padding - image_height)/2, 
                                        (image_width + min_padding - image_height)/2, 
                                        min_padding, min_padding, cv2.BORDER_CONSTANT, 
                                        value=value)

    return cv2.resize(padded_img, target_shape)


def create_2d_gaussian(dim, sigma):
    """
    This function creates a 2d gaussian kernel with the standard deviation
    denoted by sigma
    
    :param dim: integer denoting a side (1-d) of gaussian kernel
    :param sigma: floating point indicating the standard deviation
    
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
    
    return kernel/sum(sum(kernel))


def elastic_transform(image, kernel_dim=13, sigma=6, alpha=36, negated=False):
    """
    This method performs elastic transformations on an image by convolving 
    with a gaussian kernel.

    :param image: a numpy nd array
    :kernel_dim: dimension(1-D) of the gaussian kernel
    :param sigma: standard deviation of the kernel
    :param alpha: a multiplicative factor for image after convolution
    :param negated: a flag indicating whether the image is negated or not

    :returns: a nd array transformed image
    """
    # check if the image is a negated one
    if not negated:
        image = 255-image

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
            low_ii = row + int(math.floor(displacement_field_y[row, col]))
            high_ii = row + int(math.ceil(displacement_field_y[row, col]))

            low_jj = col + int(math.floor(displacement_field_y[row, col]))
            high_jj = col + int(math.ceil(displacement_field_y[row, col]))

            if low_ii < 0 or low_jj < 0 or high_ii >= image.shape[1] -1 \
               or high_jj >= image.shape[0] - 1:
                continue

            res = image[low_ii, low_jj]/4 + image[low_ii, high_jj]/4 + \
                    image[high_ii, low_jj]/4 + image[high_ii, high_jj]/4

            result[row, col] = res

    return result

        
def width_normalization(image, width, target_shape, negated=False):
    """
    This method creates a width normalised 1-d vector of an image
    
    :param image: a nd array denoting the image
    :param width: the width o which the vector shoul be normalized
    :param target_shape: a tuple denoting the output dims

    :returns: a nd array width normalized image
    """
    
    # negate the image
    if not negated:
        image = 255-image

    # crop the number bounding box
    cropped_img = do_cropping(image)

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
    return resized_image

