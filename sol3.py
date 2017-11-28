import numpy as np
import scipy.signal as sig
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy import ndimage

NUM_ROWS = 0
NUM_COLUMNS = 1
GRAY_REPR = 1
RGB_REPR = 2
RGB_SHAPE = 3
NUM_GRAY_SCALE = 255
TWO = 2
PADDING = 0

def read_image(filename, representation):
    """
    read the image and convert it to gray scale if necessary
    :param filename: the image
    :param representation: gray scale or RGB format to which the image is to be converted
    :return: A image whose values are normalized and in the given representation
    """
    if( representation < GRAY_REPR or representation > RGB_REPR):
        return -1

    #read the image
    img = imread(filename)

    #convert the RGB image to gray scale image
    if len(img.shape) == RGB_SHAPE and representation == GRAY_REPR:
        return rgb2gray(img)
    return np.divide(img.astype(np.float64), NUM_GRAY_SCALE)


def get_gaussian_kernel(kernel_size):
    '''
    Compute the gaussian kernel
    :param kernel_size: The length of the row/column of the kernel
    :return: The matrix that contain the gaussian kernel
    '''
    kernel_row = [1, 1]
    bin_vec = kernel_row.copy()
    #if the size is one return the gaussian kernel [1]
    if kernel_size == 1:
        gaussian_kernel = [1]
        return np.asarray(gaussian_kernel).reshape(1,1)

    # use convolution to achieve the binomial co-efficient
    while kernel_size != TWO:
        kernel_row = sig.convolve(kernel_row, bin_vec)
        kernel_size -=1
    # get the matrix and divide it the by the sum of the values
    kernel_row = kernel_row/kernel_row.sum()
    kernel_row = kernel_row.reshape(1,len(kernel_row))
    kernel_col = kernel_row.reshape(kernel_row.shape[1], 1)

    return kernel_row,kernel_col

def reduce_gaussian_pyramid(img,row_filter, col_filter):
    '''
    reduce the img to size of n/2 times n/2
    :param img: The last level in the pyramid to be reduced
    :param row_filter: A row gaussian kernel
    :param col_filter: A col gaussian kernel
    :return: An image of n/2 times n/2
    '''
    #blur the img
    blurred_img = ndimage.convolve(img,row_filter)
    blurred_img = ndimage.convolve(blurred_img, col_filter)
    #sub-sample every second pixel
    return blurred_img[0::2, 0::2]

def build_gaussian_pyramid(im, max_levels, filter_size):
    '''
    Build gaussian pyramid
    :param im: the original image
    :param max_levels: The number of layers in the pyramid
    :param filter_size: The gaussian filter size
    :return: The gaussian pyramid and the gaussian filter vector
    '''
    #get the gaussian filter
    row_filter, col_filter = get_gaussian_kernel(filter_size)
    pyr =[]
    pyr.append(im)
    last_level = im
    # build the pyramid i.e reduced the images
    for level in range(max_levels):
        last_level = reduce_gaussian_pyramid(last_level,row_filter, col_filter)
        pyr.append(last_level)

    return pyr, row_filter


def expand(im, row_filter):
    col_filter = row_filter.resahpe(len(row_filter),1)


def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    Build laplacian pyramid
    :param im: the original image
    :param max_levels: The number of layers in the pyramid
    :param filter_size: The gaussian filter size
    :return: The laplacian pyramid and the gaussian filter vector
    '''
    #get the gaussian filter and the gaussian pyramid
    gaussian_pyr, row_filter = build_laplacian_pyramid(im, max_levels, filter_size)


    pyr = []
    for level in range(max_levels-1):
        laplacian_level = gaussian_pyr[level] - expand(gaussian_pyr[level+1], row_filter)
        pyr.append(laplacian_level)

    return pyr, row_filter


def main():
    im = read_image("gray_orig.png", 1)
    max_levels = 2
    filter_size = 3

    x= np.array([1,2,3])

    print(np.insert(x,[::2],0))
    pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    a=3

if "__name__=__main__":
    main()
