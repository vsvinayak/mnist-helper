from mnist_helpers import *

import cv2

# this is the image we want to distort
test_image = './images//four.png'

if __name__ == '__main__':

    # read the grayscale image
    image = cv2.imread(test_image, 0)

    # just call the function elastic_transform function 
    # with a suitable kernel size, alpha and sigma
    # as a rule of thumb, if use sigma as a value near 6,
    # alpha 36-40, kernel size 13-15
    #
    # NOTE: the input image SHOULD be of square dimension,
    # ie no.of rows should be equal to number of cols.
    
    image = cv2.resize(image, (30,30))

    # get the transformed image
    distorted_image = elastic_transform(image, kernel_dim=15,
                                        alpha=5.5,
                                        sigma=35)

    cv2.imwrite('./images/distortd.png', distorted_image)
