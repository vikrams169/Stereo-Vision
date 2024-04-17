# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

# Detecting SIFT Features from an RGB image (represented as (keypoint, feature))
def detect_sift_features(rgb_img):
    gray_img = cv2.cvtColor(copy.deepcopy(rgb_img),cv2.COLOR_BGR2GRAY) # Converting the RGB image to grayscale
    sift_function = cv2.SIFT_create() # Creating an instance of the SIFT Function
    keypoints, features = sift_function.detectAndCompute(gray_img,None) # Computing the set of keypoints and features for the image
    #kp_img = cv2.drawKeypoints(gray_img,keypoints,rgb_img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Drawing the keypints/features on the grayscale image for visualization
    return keypoints, features#, kp_img

# Matching features between images using a FLANN feature matcher
def FLANN_match_features(features1,features2):
    good_matches = [] # Initializing the final list of good feature matches
    flann_matcher = cv2.FlannBasedMatcher_create() # Creating an instance of the FLANN Matcher
    matches = flann_matcher.knnMatch(features1,features2,k=2) # Finding the FLANN macthes
    # Using only the matches that conform to features being within a certain thresholded distance from eachother
    for feature1, feature2 in matches:
        if feature1.distance < 0.3*feature2.distance:
            good_matches.append(feature1)
    return good_matches

def drawlines(img1, img2, lines, pts1, pts2): 
    
    r, c = img1.shape 
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) 
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) 
      
    for r, pt1, pt2 in zip(lines, pts1, pts2): 
          
        color = tuple(np.random.randint(0, 255, 
                                        3).tolist()) 
          
        x0, y0 = map(int, [0, -r[2] / r[1] ]) 
        x1, y1 = map(int,  
                     [c, -(r[2] + r[0] * c) / r[1] ]) 
          
        img1 = cv2.line(img1,  
                        (x0, y0), (x1, y1), color, 1) 
        img1 = cv2.circle(img1, 
                          (int(pt1[0,0]),int(pt1[0,1])), 5, color, -1) 
        img2 = cv2.circle(img2,  
                          (int(pt2[0,0]),int(pt2[0,1])), 5, color, -1) 
    return img1, img2 

def draw_epipolar_lines(rgb_img1,rgb_img2,good_matches,kp1,kp2):
    img1 = copy.deepcopy(cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2GRAY))
    img2 = copy.deepcopy(cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2GRAY))
    left_points = np.array([kp1[matches.queryIdx].pt for matches in good_matches]).reshape(-1,1,2).astype(np.float32)
    right_points = np.array([kp2[matches.trainIdx].pt for matches in good_matches]).reshape(-1,1,2).astype(np.float32)
    F, mask = cv2.findFundamentalMat(left_points, 
                                 right_points, 
                                 cv2.FM_RANSAC,0.1,0.99)
    left_points = left_points[mask.ravel() == 1] 
    right_points = right_points[mask.ravel() == 1]
    linesLeft = cv2.computeCorrespondEpilines(right_points.reshape(-1, 
                                                           1, 
                                                           2), 
                                          2, F) 
    linesLeft = linesLeft.reshape(-1, 3) 
    img5, img6 = drawlines(img1, img2,  
                        linesLeft, left_points, 
                        right_points) 
    
    # Find epilines corresponding to  
    # points in left image (first image) and 
    # drawing its lines on right image 
    linesRight = cv2.computeCorrespondEpilines(left_points.reshape(-1, 1, 2),  
                                            1, F) 
    linesRight = linesRight.reshape(-1, 3) 
    
    img3, img4 = drawlines(img1, img2,  
                        linesRight, right_points, 
                        left_points) 
    
    plt.subplot(121), plt.imshow(img5) 
    plt.subplot(122), plt.imshow(img3) 
    plt.show()

def getX(line, y):
    x = -(line[1]*y + line[2])/line[0]
    return x

def getEpipolarLines(set1, set2, F, image0, image1, rectified = False):
    # set1, set2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]
    lines1, lines2 = [], []
    img_epi1 = image0.copy()
    img_epi2 = image1.copy()

    for i in range(set1.shape[0]):
        x1 = np.array([set1[i,0], set1[i,1], 1]).reshape(3,1)
        x2 = np.array([set2[i,0], set2[i,1], 1]).reshape(3,1)

        line2 = np.dot(F, x1)
        lines2.append(line2)

        line1 = np.dot(F.T, x2)
        lines1.append(line1)
    
        if not rectified:
            y2_min = 0
            y2_max = image1.shape[0]
            x2_min = getX(line2, y2_min)
            x2_max = getX(line2, y2_max)

            y1_min = 0
            y1_max = image0.shape[0]
            x1_min = getX(line1, y1_min)
            x1_max = getX(line1, y1_max)
        else:
            x2_min = 0
            x2_max = image1.shape[1] - 1
            y2_min = -line2[2]/line2[1]
            y2_max = -line2[2]/line2[1]

            x1_min = 0
            x1_max = image0.shape[1] -1
            y1_min = -line1[2]/line1[1]
            y1_max = -line1[2]/line1[1]



        cv2.circle(img_epi2, (int(set2[i,0]),int(set2[i,1])), 10, (0,0,255), -1)
        img_epi2 = cv2.line(img_epi2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 0, int(i*2.55)), 2)
    

        cv2.circle(img_epi1, (int(set1[i,0]),int(set1[i,1])), 10, (0,0,255), -1)
        img_epi1 = cv2.line(img_epi1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 0, int(i*2.55)), 2)

    plt.subplot(121), plt.imshow(img_epi1) 
    plt.subplot(122), plt.imshow(img_epi2) 
    plt.show() 

def unrectified_image_association(good_matches,kp1,kp2,K1,K2):
    img1_points = np.array([kp1[matches.queryIdx].pt for matches in good_matches]).reshape(-1,1,2).astype(np.float32)
    img2_points = np.array([kp2[matches.trainIdx].pt for matches in good_matches]).reshape(-1,1,2).astype(np.float32)
    fundamental_matrix, mask = cv2.findFundamentalMat(img1_points,img2_points,cv2.FM_RANSAC,0.1,0.99)
    essential_matrix = np.dot(K1.T,np.dot(fundamental_matrix,K2))
    U, S, V = np.linalg.svd(essential_matrix)
    #W = np.array([[0,-1,0],[1,0,0],[0,0,1]]).astype(np.float32)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    R = np.dot(U,np.dot(W,V))
    T = U[:,2]
    print("The Fundamental Matrix is computed to be: ")
    print(fundamental_matrix)
    print("The Essential Matrix is computed to be: ")
    print(essential_matrix)
    print("The Rotation Matrix computed from the Essential Matrix is: ")
    print(R)
    print("The Translation Vector computed from the essential Matrix is: ")
    print(T)
    return fundamental_matrix, essential_matrix, R, T

def rectify_stereo_pair(img1,img2,R,T,K1,d1,K2,d2):
    H1, H2, P1, P2, Q, roi_1, roi_2 = cv2.stereoRectify(K1,d1,K2,d2,[img1.shape[1],img1.shape[0]],R,T,alpha=1.0)
    print("The homography/rotation matrix (H1) for rectifying the first/left image: ")
    print(H1)
    print("The homography/rotation matrix (H2) for rectifying the second/right image: ")
    print(H2)
    map_x_img1, map_y_img1 = cv2.initUndistortRectifyMap(K1,d1,H1,K1,[img1.shape[1],img1.shape[0]],cv2.CV_32F)
    map_x_img2, map_y_img2 = cv2.initUndistortRectifyMap(K2,d2,H2,K2,[img2.shape[1],img2.shape[0]],cv2.CV_32F)
    img1_rect = cv2.remap(img1,map_x_img1,map_y_img1,cv2.INTER_LINEAR)
    img2_rect = cv2.remap(img2,map_x_img2,map_y_img2,cv2.INTER_LINEAR)
    # Use cv2.INTER_CUBIC above instead
    img_concat = np.concatenate([copy.deepcopy(img1_rect),copy.deepcopy(img2_rect)],axis=1)
    img_concat = cv2.cvtColor(img_concat,cv2.COLOR_BGR2RGB)
    plt.imshow(img_concat)
    plt.show()
    return img1_rect, img2_rect

def generate_disparity_map(img1_rect,img2_rect,window_size=11):
    imgL = cv2.resize(copy.deepcopy(img1_rect),(int(img1_rect.shape[1]/4), int(img1_rect.shape[0]/4)))
    imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY).astype(np.int32)
    imgR = cv2.resize(copy.deepcopy(img2_rect),(int(img2_rect.shape[1]/4), int(img2_rect.shape[0]/4)))
    imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY).astype(np.int32)
    h, w = imgL.shape
    disparity_map = np.zeros((h,w))
    disp_block_width = w - 2*window_size
    print("Generating the Disparity Map. This may take a couple seconds.")
    for i in range(window_size,h-window_size):
        imgL_blocks = []
        imgR_blocks = []
        for j in range(window_size,w-window_size):
            left_img_block = imgL[i:i+window_size,j:j+window_size]
            right_img_block = imgR[i:i+window_size,j:j+window_size]
            imgL_blocks.append(left_img_block.flatten())
            imgR_blocks.append(right_img_block.flatten())
        imgL_blocks = np.array(imgL_blocks)
        imgR_blocks = np.array(imgR_blocks)
        imgL_blocks = np.repeat(imgL_blocks[:,:,np.newaxis],disp_block_width,axis=2)
        imgR_blocks = np.repeat(imgR_blocks[:,:,np.newaxis],disp_block_width,axis=2).T
        blocks_diff = np.sum(np.abs(imgL_blocks-imgR_blocks),axis=1)
        ideal_idx = np.argmin(blocks_diff,axis=0)
        disparity_block = np.abs(ideal_idx - np.linspace(0,disp_block_width,disp_block_width,dtype=int)).reshape(1,disp_block_width)
        disparity_map[i,0:disp_block_width] = disparity_block 
    disparity_map = ((disparity_map/disparity_map.max())*255).astype(np.int32)
    #disparity_map = np.uint8(disparity_map * 255 / np.max(disparity_map))
    print("The Disparity Map using a Heatmap Visualization: ")
    plt.imshow(disparity_map,cmap='hot',interpolation='nearest')
    plt.show()
    print("The Disparity Map using a Grayscale Visualization: ")
    plt.imshow(disparity_map,cmap='gray',interpolation='nearest')
    plt.show()
    return disparity_map

def generate_depth_map(disparity_map,baseline,focal_length):
    depth_map = (baseline*focal_length)/(disparity_map+np.exp(-10))
    depth_map[depth_map>100000] = 100000
    #depth_map = np.uint8(depth_map * 255 / np.max(depth_map))
    depth_map = ((depth_map/depth_map.max())*255).astype(np.int32)
    print("The Depth Map using a Heatmap Visualization: ")
    plt.imshow(depth_map,cmap='hot',interpolation='nearest')
    plt.show()
    print("The Depth Map using a Grayscale Visualization: ")
    plt.imshow(depth_map,cmap='gray',interpolation='nearest')
    plt.show()
    return depth_map

def stereo_vision_wrapper(img1,img2,K1,d1,K2,d2,baseline):
    # The line below is valid only when K1 == K2 and fx == fy
    focal_length = K1[0,0]
    rgb_img1 = copy.deepcopy(img1)
    rgb_img2 = copy.deepcopy(img2)
    kp1, features1 = detect_sift_features(copy.deepcopy(rgb_img1))
    kp2, features2 = detect_sift_features(copy.deepcopy(rgb_img2))
    good_matches = FLANN_match_features(features1,features2)
    F, E, R, T = unrectified_image_association(good_matches,kp1,kp2,K1,K2)
    draw_epipolar_lines(img1,img2,good_matches,kp1,kp2)
    img1_rect, img2_rect = rectify_stereo_pair(rgb_img1,rgb_img2,R,T,K1,d1,K2,d2)
    kp1, features1 = detect_sift_features(copy.deepcopy(img1_rect))
    kp2, features2 = detect_sift_features(copy.deepcopy(img2_rect))
    good_matches = FLANN_match_features(features1,features2)
    draw_epipolar_lines(img1_rect,img2_rect,good_matches,kp1,kp2)
    disparity_map = generate_disparity_map(img1_rect,img2_rect)
    depth_map = generate_depth_map(disparity_map,baseline,focal_length)

def main():
    img1 = cv2.imread("data/storageroom/im1.png")
    img2 = cv2.imread("data/storageroom/im0.png")
    #K1 = np.array([[1746.24,0,14.88],[0,1746.24,534.11],[0,0,1]])
    #K2 = np.array([[1746.24,0,14.88],[0,1746.24,534.11],[0,0,1]])
    K1 = np.array([[1742.11,0,804.90],[0,1742.11,541.22],[0,0,1]])
    K2 = np.array([[1742.11,0,804.90],[0,1742.11,541.22],[0,0,1]])
    d1 = np.array([0,0,0,0]).astype(np.float32)
    d2 = np.array([0,0,0,0]).astype(np.float32)
    baseline = 221.76
    stereo_vision_wrapper(img1,img2,K1,d1,K2,d2,baseline)

if __name__ == "__main__":
    main()


























