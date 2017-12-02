import numpy as np
import scipy.signal as sig
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy import ndimage
import matplotlib.pyplot as plt

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
    for level in range(max_levels-1):#todo check about the max level size
        if not (last_level.shape[0] == 16) or not (last_level.shape[1] == 16):
            last_level = reduce_gaussian_pyramid(last_level, row_filter, col_filter)
            pyr.append(last_level)

    return pyr, row_filter


def expand(im, row_filter):
    '''
    Expend the img to size of n*2 times n*2
    :param im: The last level in the pyramid to be expended
    :param row_filter: A row gaussian kernel
    :return: An image of n*2 times n*2
    '''
    row_filter *= TWO
    col_filter = row_filter.reshape(row_filter.shape[1], 1)
    #Create the img matrix and assign the smaller img on the odo pixels
    expanded_im = np.zeros((im.shape[0]*TWO, im.shape[1]*TWO))
    expanded_im[1::2, 1::2] = im
    #Convolve with a gaussian kernel
    expanded_im = ndimage.convolve(expanded_im, row_filter)
    expanded_im = ndimage.convolve(expanded_im, col_filter)
    return expanded_im


def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    Build laplacian pyramid
    :param im: the original image
    :param max_levels: The number of layers in the pyramid
    :param filter_size: The gaussian filter size
    :return: The laplacian pyramid and the gaussian filter vector
    '''
    #get the gaussian filter and the gaussian pyramid
    gaussian_pyr, row_filter = build_gaussian_pyramid(im, max_levels, filter_size)

    #build the laplacain pyramid
    pyr = []
    for level in range(len(gaussian_pyr)-1):
        laplacian_level = gaussian_pyr[level] - expand(gaussian_pyr[level+1], row_filter.copy())
        pyr.append(laplacian_level)
    #The top of the pyramid is the same of the top of the gaussian pyramid
    pyr.append(gaussian_pyr[len(gaussian_pyr)-1])

    return pyr, row_filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    '''
    Reconstruct the image from the laplacian pyramid
    :param lpyr: The laplacian pyramid
    :param filter_vec: The gaussian kernel
    :param coeff: A vector of coefficient
    :return: The reconstructed image
    '''
    #Multiply the laplacian pyramid by their coefficient
    coeff = np.asarray(coeff)
    lpyr = coeff*lpyr
    #Reconstruct the image
    re_img = lpyr[len(lpyr)-1]
    for level in range((len(lpyr) - 2), -1, -1):
        re_img = lpyr[level] + expand(re_img, filter_vec)

    return re_img


def render_pyramid(pyr, levels):
    '''
    Render the pyramid into one image
    :param pyr: The gaussian/laplapcian pyramid
    :param levels: The number of levels to be seen
    :return: An image that consist of all the pyramid's images
    '''
    row,col = (0, 0)
    pixel = 0
    #scaling //todo not sure about this
    for i in range(levels):
        pyr[i] = (pyr[i] - pyr[i].min())/(pyr[i].max() - pyr[i].min())
    #figure out the size of the image
    for index in range(levels):
        row += pyr[index].shape[0]
        col += pyr[index].shape[1]

    #assigning the gaussian/laplacian pyramid into one image
    pyr_im = np.zeros((row, col))
    for j in range(levels): #todo check about the max levels
        pyr_im[0:pyr[j].shape[0]:1, pixel:pixel+pyr[j].shape[1]:1] = pyr[j]
        pixel += pyr[j].shape[1]

    return pyr_im


def display_pyramid(pyr, levels):
    '''
    display the gaussian/laplacian pyramid
    :param pyr: the gaussian/laplacian pyramid
    :param levels: the number of level to be display
    '''

    #create one image for all the pyramid levels
    pyr_im = render_pyramid(pyr, levels)
    plt.imshow(pyr_im, cmap=plt.cm.gray)
    plt.show()


def build_blend_pyramid(lap_pyr1, lap_pyr2, mask_pyr, max_levels):
    '''
    Build the blended image pyramid
    :param lap_pyr1: The laplacian pyramid of the first image
    :param lap_pyr2: The laplacian pyramid of the second image
    :param mask_pyr: The laplacian pyramid of the mask
    :param max_levels: The number of level in the pyramid
    :return: The blended pyramid
    '''
    lap_blend = []
    # Creating the blended image pyramid levels
    for level in range(max_levels):
        blend_level = (mask_pyr[level] * lap_pyr1[level]) + ((1 - mask_pyr[level]) * lap_pyr2[level])
        lap_blend.append(blend_level)
    return lap_blend

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    '''
    Blend two images via constructing their laplacian pyramid
    :param im1: The first image
    :param im2: The second image
    :param mask: A mask that consist of boolean values
    :param max_levels: The number of the level in the laplacian pyramids
    :param filter_size_im: The size of the gaussian filter to be used upon creating the laplacian pyramid of the images
    :param filter_size_mask: The size of the gaussian filter to be used upon creating the laplacian pyramid of the mask
    :return: A single blended image of the original images size
    '''
    #Building the laplacian pyramid of the images
    lap_pyr1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap_pyr2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)

    #Building the gaussian pyramid of the mask
    mask = np.asarray(mask).astype(np.float64)
    mask_pyr, filter_mask = build_gaussian_pyramid(mask, max_levels, filter_size_mask)

    #Creating the blended image pyramid
    lap_blend = build_blend_pyramid(lap_pyr1, lap_pyr2, mask_pyr, max_levels)

    #reconstruct the blended image from the pyramid and clip the values to [0,1]
    coeff = np.ones(len(lap_blend))
    return np.clip(laplacian_to_image(lap_blend, filter_size_im, coeff), 0, 1)

def main():
    im = read_image("gray_orig.png", 1)
    max_levels = 4
    filter_size = 3

    # x= np.array([[1,2,3],[4,5,6]])
    # y=np.array([1,2]).reshape(2,1)
    # print(x*y)
    pyr, filter_vec = build_laplacian_pyramid(im, max_levels, filter_size)
    #pyr_im = display_pyramid(pyr,4)
    img = laplacian_to_image(pyr,filter_vec, [1,1,1,1])

if "__name__=__main__":
    main()
