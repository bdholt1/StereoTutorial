import numpy as np
import cv2

def SAD(left_block, right_block):
    """Sum of absolute differences"""
    return np.sum(np.abs(left_block - right_block))

def SSD(left_block, right_block):
    """Sum of squared differences"""
    return np.sum( (left_block - right_block)**2 )

def block_match(left, right, blocksize=11, max_disp=50):
    """
    Perform basic block matching on left and right rectified images.

    Assume that the images have ben undistorted and rectified.

    :param left The left image
    :param right The right image
    :param blocksize The size of the block for matching
    :param max_disp The maximum range to search over
    """

    # OpenCV created matrices of type uint8 for 8-bit grayscale
    # Need to convert them to floats to avoid overflow in SAD/SSD operation
    left_float = left.astype(np.float32)
    right_float = right.astype(np.float32)
    h, w = left.shape
    disp = np.zeros((h, w), np.uint8)

    offset_adjust = 255. / max_disp
    m = blocksize//2
    for row in range(m, h-m):
        for col in range(m, w-m):
            # this is block we're matching against
            right_block = right_float[row-m:row+m+1, col-m:col+m+1]
            best_cost = 1e10
            best_d = -1
            # consider a range of possible disparities (only positive)
            for d in range(0, max_disp+1):
                # do range checking on the block
                if col-m+d < 0 or col+m+d >= w:
                    continue
                left_block = left_float[row-m:row+m+1, col-m+d:col+m+d+1]
                cost = SAD(left_block, right_block)
                if cost < best_cost:
                    best_cost = cost
                    best_d = d
            if best_d != -1:
                disp[row, col] = best_d * offset_adjust
    return disp


left = cv2.imread('left.png')
right = cv2.imread('right.png')

left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

cv2.imshow('left',  left_gray)
cv2.imshow('right',  right_gray)

disparity = block_match(left_gray, right_gray)
disp_color = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
cv2.imshow('disparity',  disp_color)
cv2.waitKey(-1)
cv2.imwrite('disparity_basic.png', disp_color)
