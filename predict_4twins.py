'''
Copyright is preserved to Quoc-Tin Phan (dimmoon2511[at]gmail.com)
'''

import cv2 as cv
import numpy as np 
from utils import *
from data_decoder import *
import siamese_net_couple
import net_solver

def estimate_trans_from_mask_andrea(im, mask, isPlot=False):
    '''
    This function estimates the transformation from binary mask
    Modified by Andrea
    Input:
        im: the RGB image
        mask: binary mask
    Return:
        3x3 transformation matrix,
        None if the mask do not obey (1-1) condition
    '''
    h,w = mask.shape[0],mask.shape[1]
    # preprocessing
    _,mask_res = cv.threshold(mask, 0.5, 1.0, cv.THRESH_BINARY)
    # remove dots
    # element = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    # mask_res = cv.erode(mask_res, element, iterations = 1)
    # mask_res = cv.dilate(mask_res, element, iterations = 1)
    
    # num_labels,labels = cv.connectedComponents(np.uint8(mask_res))
    # if num_labels == 3:

    # Removing dots: if localisation is accurate (sometimes even too much), opt for stronger erosion (do not dilate).
    # This will get rid of (most of) the problematic regions
    element = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    mask_res = cv.erode(mask_res, element, iterations = 1)
    mask_res = cv.dilate(mask_res, element, iterations = 1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask_res), 4, cv2.CV_32S)

    # Less than 3 regions: give up disambiguation
    if num_labels < 3:
        return None, None, None

    # Three or more regions
    elif num_labels >= 3:
        # Sorting components by area, larger to smaller
        sorted_by_area = np.sort(stats[:, 4])[::-1]

        # Ratio between third largest area and 4th, 5th and so on areas.
        r_threshold = 0.8
        ratio_areas = stats[:, 4] / sorted_by_area[2]

        # If ratio is above an empirical threshold just for three regions, then continue disambiguating: force to 0
        # non-interesting regions in labelled mask
        labels2 = labels.copy()
        if np.sum(ratio_areas > r_threshold) == 3:
            num_labels = 3
            cnt = 0
            # Re-label mask with [0, 1, 2], assign small regions to background (assuming always label=0 for bkg?)
            for i, t in enumerate(ratio_areas > r_threshold):
                if t:
                    labels2[labels == i] = cnt
                    cnt += 1
                else:
                    labels2[labels == i] = 0

            labels = labels2

        # Third and 4th region (at least) are too similar in size to each other, giving up after all
        else:
            return None, None, None

        means = np.zeros((2,2))
        pc = np.zeros((2,2))
        mask_pc = cv.cvtColor(mask_res, cv.COLOR_GRAY2RGB)
        cov_zero = False
        for l in range(1,num_labels):
            # extract points from each cluster
            seg_mask = labels==l
            if np.sum(seg_mask) < 32:
                #print('region %d has less than 32 pixels' % l)
                return None, None, None
            y_coords, x_coords = np.nonzero(seg_mask)
            coords = np.column_stack([x_coords, y_coords])
            coords_, mu = mean_normalize(coords)
            #coords_, _ = std_normalize(coords_)
            means[:,l-1] = mu
            # find principle component
            cov = np.cov(coords_, rowvar=False)
            if np.abs(cov[0,1]) > 1e-6: # check if cov of x,y coordinates are zero
                U,S,V = np.linalg.svd(cov)
                u1 = U[:,0]
                pc[:,l-1] = u1
            else:
                cov_zero = True

        # rotate
        if not cov_zero:
            alphas = np.arctan2(pc[1,:], pc[0,:])
            alpha = alphas[1] - alphas[0]
        else:
            #print('===== ZERO COV: %f, %f =====' % (cov[0,1], cov[1,0]))
            alpha = 2.0*np.pi/180.0

        T_rot = np.array([[np.cos(alpha), -np.sin(alpha), 0], \
                         [np.sin(alpha), np.cos(alpha), 0], \
                         [0, 0, 1]])
        y_coords_1, x_coords_1 = np.nonzero(labels==1)
        coords_1 = np.column_stack([x_coords_1, y_coords_1])
        y_coords_2, x_coords_2 = np.nonzero(labels==2)
        coords_2 = np.column_stack([x_coords_2, y_coords_2])
        coords_1_t = cv.transform(coords_1[:,np.newaxis,:], T_rot[:2,:])
        coords_1_t = coords_1_t.squeeze()

        # rescaling
        min_x, max_x = coords_1_t[:,0].min(), coords_1_t[:,0].max()
        min_y, max_y = coords_1_t[:,1].min(), coords_1_t[:,1].max()
        sc_x = (x_coords_2.max() - x_coords_2.min()) / (max_x - min_x)
        sc_y = (y_coords_2.max() - y_coords_2.min()) / (max_y - min_y)
        T_sc = np.eye(3)
        T_sc[0,0] = sc_x
        T_sc[1,1] = sc_y
        coords_1_t = cv.transform(coords_1_t[:,np.newaxis,:], T_sc[:2,:])
        coords_1_t = coords_1_t.squeeze()

        # need to update mean after rescaling
        means[:,0] = coords_1_t.mean(axis=0).T

        # compute translation
        T_tran = np.eye(3)
        T_tran[:2,2] = (means[:,1] - means[:,0])
        coords_1_t = cv.transform(coords_1_t[:,np.newaxis,:], T_tran[:2,:])
        coords_1_t = coords_1_t.squeeze()

        # integrate all transformations
        T = T_tran.dot(T_sc.dot(T_rot))
        return T, coords_1.T, coords_2.T # transform coords_1 to coords_2 by T

def predict_with_mask(predictor, im, mask):
    '''
    This function performs prediction given image and binary mask
    Input:
        predictor: pretrained model
        im: RGB image
        mask: binary mask
    Output:
        RGB mask,
        None if the transformation matrix is not successfully estimated
    '''
    h,w = im.shape[0],im.shape[1]
    assert h==mask.shape[0] and w==mask.shape[1], 'dimensions of image and mask are mismatched'
    if mask.ndim > 2:
        mask = 255-mask[...,-1] # use copy-move mask only
    
    T, src_coords, dst_coords = estimate_trans_from_mask_andrea(im, mask)
    
    if T is not None:# and not is_rigid(T): # having two disconnected regions + not rigid
        I_dst, I_dst_est, I_src, I_src_est = compute_pair_from_coords(im, T, src_coords, dst_coords)

        output_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # predict background
        output_mask[:,:,2] = 255
        output_mask[src_coords[1,:], src_coords[0,:], 2] = 0
        output_mask[dst_coords[1,:], dst_coords[0,:], 2] = 0
        soft_pred,pred_cls = predictor.predict_couple(I_dst/255, I_dst_est/255, I_src/255, I_src_est/255)
        
        # if pred_cls[0]==1, red:target, green:source
        # if pred_cls[0]==0, red:source, green:target
        pred_cls = pred_cls.squeeze()
        output_mask[src_coords[1,:], src_coords[0,:], int(pred_cls[0])] = 255
        output_mask[dst_coords[1,:], dst_coords[0,:], 1-int(pred_cls[0])] = 255
        
        return output_mask
    else:
        return None

if __name__ == "__main__":
    net = siamese_net_couple.initialize({
        'use_tf_threading'  :   False,
        'train_runner'      :   None
    })
    net.model()
    predictor = net_solver.initialize({
        'working_dir'   :   'pretrained_4twins'
    })
    predictor.setup_net(net, ckpt_id=375000)
    
    # to be configured
    images = ['resources/test01.tif', 'resources/test02.tif', 'resources/test03.tif']
    masks = ['resources/test01_tm.png', 'resources/test02_tm.png', 'resources/test03_tm.png']
    for img_path,mask_path in zip(images, masks):
        image       = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        mask        = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        pred_mask   = predict_with_mask(predictor, image, mask)
        if pred_mask is not None:
            pred_mask = cv.cvtColor(pred_mask.astype(np.uint8), cv.COLOR_RGB2BGR)
            cv.imwrite(mask_path.replace('_tm', '_4twins_pred'), pred_mask)
        else:
            print('Tampering map does not obey (1-1) condition')