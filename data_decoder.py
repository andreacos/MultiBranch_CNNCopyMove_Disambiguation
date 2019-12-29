'''
Copyright is preserved to Quoc-Tin Phan (dimmoon2511[at]gmail.com)
'''
import numpy as np
import cv2
from utils import *

def compute_pair(I, cmd_mask, trans_mat, is_crop=True):
    '''
    This function extracts two pairs from RGB mask
    Input:
        I: RGB image
        cmd_mask: RGB mask
        R=Target, G=Source, B=Background
        trans_mat: transformation matrix from source to destination
        is_crop: if true, return center-cropped regions of size 64x64x3
    Output: destination, transform of source, source, transform of destination
    '''
    dst_coords_y,dst_coords_x = np.nonzero(cmd_mask[:,:,0])
    src_coords_y,src_coords_x = np.nonzero(cmd_mask[:,:,1])
    dst_coords = np.stack((dst_coords_x, dst_coords_y))
    src_coords = np.stack((src_coords_x, src_coords_y))
    # forward & backward transform
    I_dst = reconstruct(I, dst_coords_x, dst_coords_y)
    I_src = reconstruct(I, src_coords_x, src_coords_y)
    # get a pair
    I_dst_est, I_src_est = get_pairs_est(I, src_coords, dst_coords, trans_mat)
    if is_crop:
        return  center_crop(I_dst), center_crop(I_dst_est),\
                center_crop(I_src), center_crop(I_src_est)
    else:
        return I_dst, I_dst_est, I_src, I_src_est

def compute_pair_from_coords(I, trans_mat, src_coords, dst_coords, is_crop=True, crop_position='center'):
    '''
    This function extracts two pairs from dense coordinates
    Input:
        I: RGB imaged
        trans_mat: transformation matrix from source to destination
        src_coords: 2xn matrix, 2D coordinates of n points in source region
        dst_coords: 2xn matrix, 2D coordinates of n points in destination region
        is_crop: if true, return center-cropped regions of size 64x64x3
        crop_position: 'center' (default), 'corner' (5 corners), 'multiple' (5 corners and center)
    Output: destination, transform of source, source, transform of destination
    '''
    # forward & backward transform
    I_dst = reconstruct(I, dst_coords[0,:], dst_coords[1,:])
    I_src = reconstruct(I, src_coords[0,:], src_coords[1,:])
    # get a pair
    I_dst_est, I_src_est = get_pairs_est(I, src_coords, dst_coords, trans_mat)
    if is_crop:
        if crop_position == 'corner':
            return  corner_crop(I_dst), corner_crop(I_dst_est),\
                    corner_crop(I_src), corner_crop(I_src_est)
        elif crop_position == 'multiple':
            return  multiple_crop(I_dst), multiple_crop(I_dst_est),\
                    multiple_crop(I_src), multiple_crop(I_src_est)
        else:
            return  center_crop(I_dst), center_crop(I_dst_est),\
                    center_crop(I_src), center_crop(I_src_est)
    else:
        return I_dst, I_dst_est, I_src, I_src_est

def reconstruct(I, coords_2d_x, coords_2d_y):
    '''
    This function extracts the bounding region given coordinates
    Input:
        I: RGB image
        coords_2d_x: n dimensional vector of x coordinates of n points
        coords_2d_y: n dimensional vector of y coordinates of n points
    Output: the extracted region
    '''
    tl = (np.min(coords_2d_x), np.min(coords_2d_y))
    br = (np.max(coords_2d_x), np.max(coords_2d_y))
    w,h = (br[0] - tl[0], br[1] - tl[1])
    x = np.linspace(tl[0], br[0]-1,w).astype(np.int)
    y = np.linspace(tl[1], br[1]-1,h).astype(np.int)
    xv,yv = np.meshgrid(x, y)
    I_sub = I[yv, xv, :]
    return I_sub

def get_pairs_est(I, src_coords, dst_coords, trans_mat):
    '''
    This function performs forward and backward transforms
    to get the estimate of destination and source regions,
    Bilinear interpolation is used
    Input: 
        I: RGB image
        src_coords: 2xn matrix, 2D coordinates of n points in source region
        dst_coords: 2xn matrix, 2D coordinates of n points in destination region
        trans_mat: transformation matrix from source to destination
    '''
    trans_mat_inv = np.linalg.inv(trans_mat)

    # get full coords of dst region
    tl = (np.min(dst_coords[0,:]), np.min(dst_coords[1,:]))
    br = (np.max(dst_coords[0,:]), np.max(dst_coords[1,:]))
    w,h = (br[0] - tl[0], br[1] - tl[1])
    x = np.linspace(tl[0], br[0]-1,w)
    y = np.linspace(tl[1], br[1]-1,h)
    xv,yv = np.meshgrid(x, y)
    dst_shape = xv.shape
    dst_coords_est_full = np.stack((xv.reshape((-1)), yv.reshape((-1))))

    # get full coords of src region
    tl = (np.min(src_coords[0,:]), np.min(src_coords[1,:]))
    br = (np.max(src_coords[0,:]), np.max(src_coords[1,:]))
    w,h = (br[0] - tl[0], br[1] - tl[1])
    x = np.linspace(tl[0], br[0]-1,w)
    y = np.linspace(tl[1], br[1]-1,h)
    xv,yv = np.meshgrid(x, y)
    src_shape = xv.shape
    src_coords_est_full = np.stack((xv.reshape((-1)), yv.reshape((-1))))
    
    # compute I_dst_est
    srt_coords_est = cv2.transform(dst_coords_est_full.T[:,np.newaxis,:], trans_mat_inv[:2,:])
    srt_coords_est = np.squeeze(srt_coords_est).T 
    mapX = np.reshape(srt_coords_est[0,:], dst_shape)
    mapY = np.reshape(srt_coords_est[1,:], dst_shape)
    dstMap1, dstMap2 = cv2.convertMaps(mapX.astype(np.float32),\
                                        mapY.astype(np.float32), cv2.CV_16SC2)
    I_dst_est = np.zeros((mapX.shape[0], mapX.shape[1], I.shape[2]))
    for c in range(I.shape[2]):
        I_dst_est[:,:,c] = cv2.remap(I[:,:,c], dstMap1, dstMap2, cv2.INTER_LINEAR)
    
    # compute I_src_est
    dst_coords_est = cv2.transform(src_coords_est_full.T[:,np.newaxis,:], trans_mat[:2,:])
    dst_coords_est = np.squeeze(dst_coords_est).T 
    mapX = np.reshape(dst_coords_est[0,:], src_shape)
    mapY = np.reshape(dst_coords_est[1,:], src_shape)
    dstMap1, dstMap2 = cv2.convertMaps(mapX.astype(np.float32),\
                                        mapY.astype(np.float32), cv2.CV_16SC2)
    I_src_est = np.zeros((mapX.shape[0], mapX.shape[1], I.shape[2]))
    for c in range(I.shape[2]):
        I_src_est[:,:,c] = cv2.remap(I[:,:,c], dstMap1, dstMap2, cv2.INTER_LINEAR)

    return I_dst_est.astype(dtype=np.uint8), I_src_est.astype(dtype=np.uint8)

def center_crop(I, crop_size=64):
    '''
    This function performs center cropping, 0 padding if the image size
    is smaller than crop size
    Input:
        I: RGB image
        crop_size: dimension of one side, same to the remaining side
    Output: RGB image of size crop_size x crop_size x 3
    '''
    h,w,_ = I.shape
    if crop_size > min(h,w):
        # perform resizing
        f = (crop_size+2)/min([h,w])
        I = cv2.resize(I, (0,0), fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        h,w,_ = I.shape
    sx = int(np.floor(w/2) - np.floor(crop_size/2))
    sy = int(np.floor(h/2) - np.floor(crop_size/2))
    I = I[sy:sy+crop_size, sx:sx+crop_size, :]
    return I

def multiple_crop(I, crop_size=64):
    '''
    This function performs cropping at multiple positions
    Input:
        I: RGB image
        crop_size: dimension of one side, same to the remaining side
    Output: RGB image of size 5 x crop_size x crop_size x 3
    '''
    h,w,c = I.shape
    if crop_size > min(h,w):
        # perform resizing
        f = (crop_size+2)/min([h,w])
        I = cv2.resize(I, (0,0), fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        h,w,c = I.shape
    I_out = np.zeros((5,crop_size,crop_size,c), dtype=I.dtype)
    for i,p in enumerate([(0,0),(0,w-crop_size),(h-crop_size,0),(h-crop_size,w-crop_size)]):
        I_out[i,:,:,:] = I[p[0]:p[0]+crop_size, p[1]:p[1]+crop_size,:]
    sx = int(np.floor(w/2) - np.floor(crop_size/2))
    sy = int(np.floor(h/2) - np.floor(crop_size/2))
    I_out[-1,:,:,:] = I[sy:sy+crop_size, sx:sx+crop_size, :]
    return I_out

def corner_crop(I, crop_size=64):
    '''
    This function performs cropping at multiple positions: 4 corners and center
    Input:
        I: RGB image
        crop_size: dimension of one side, same to the remaining side
    Output: RGB image of size 5 x crop_size x crop_size x 3
    '''
    h,w,c = I.shape
    if crop_size > min(h,w):
        # perform resizing
        f = (crop_size+2)/min([h,w])
        I = cv2.resize(I, (0,0), fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        h,w,c = I.shape
    I_out = np.zeros((4,crop_size,crop_size,c), dtype=I.dtype)
    for i,p in enumerate([(0,0),(0,w-crop_size),(h-crop_size,0),(h-crop_size,w-crop_size)]):
        I_out[i,:,:,:] = I[p[0]:p[0]+crop_size, p[1]:p[1]+crop_size,:]
    return I_out