# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

# The first dataset that will be tested on
dataset1 = {"img1_path":"data/classroom/im0.png",
            "img2_path":"data/classroom/im1.png",
            "K1":np.array([[1746.24,0,14.88],[0,1746.24,534.11],[0,0,1]]).astype(np.float32),
            "K2":np.array([[1746.24,0,14.88],[0,1746.24,534.11],[0,0,1]]).astype(np.float32),
            "baseline":678.37}

# The second dataset that will be tested on
dataset2 = {"img1_path":"data/storageroom/im0.png",
            "img2_path":"data/storageroom/im1.png",
            "K1":np.array([[1742.11,0,804.90],[0,1742.11,541.22],[0,0,1]]).astype(np.float32),
            "K2":np.array([[1742.11,0,804.90],[0,1742.11,541.22],[0,0,1]]).astype(np.float32),
            "baseline":221.76}

# The third dataset that will be tested on
dataset3 = {"img1_path":"data/traproom/im0.png",
            "img2_path":"data/traproom/im1.png",
            "K1":np.array([[1769.02,0,1271.89],[0,1769.02,527.17],[0,0,1]]).astype(np.float32),
            "K2":np.array([[1769.02,0,1271.89],[0,1769.02,527.17],[0,0,1]]).astype(np.float32),
            "baseline":295.44}

# Combining the three datasets together into a single list
datasets = [dataset1, dataset2, dataset3]

# Detecting SIFT Features from an RGB image (represented as (keypoint, feature))
def detect_sift_features(rgb_img):
    gray_img = cv2.cvtColor(copy.deepcopy(rgb_img),cv2.COLOR_BGR2GRAY) # Converting the RGB image to grayscale
    sift_function = cv2.SIFT_create() # Creating an instance of the SIFT Function
    keypoints, features = sift_function.detectAndCompute(gray_img,None) # Computing the set of keypoints and features for the image
    return keypoints, features

# Matching features between images using a FLANN feature matcher
def FLANN_match_features(features1,features2,matching_threshold=0.3):
    good_matches = [] # Initializing the final list of good feature matches
    flann_matcher = cv2.FlannBasedMatcher_create() # Creating an instance of the FLANN Matcher
    matches = flann_matcher.knnMatch(features1,features2,k=2) # Finding the FLANN macthes
    # Using only the matches that conform to features being within a certain thresholded distance from eachother
    for feature1, feature2 in matches:
        if feature1.distance < matching_threshold*feature2.distance:
            good_matches.append(feature1)
    return good_matches

# A helper function to draw the epipolar lines on the images
def draw_lines_on_img(img1,img2,left_points,right_points,lines): 
    # Obtaining image dimensionns and converting the images to grayscale for better visualization
    h, w = img1.shape 
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR) 
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    # Iterating through all the lines/point pairs and drawing the epipolar lines  
    for r, pt1, pt2 in zip(lines, left_points, right_points):  
        line_color = tuple(np.random.randint(0,255,3).tolist()) 
        x0, y0 = map(int, [0, -r[2]/r[1]]) 
        x1, y1 = map(int,  [w, -(r[2] + r[0]*w)/r[1]]) 
        img1 = cv2.line(img1,(x0,y0),(x1,y1),line_color,1) 
        img1 = cv2.circle(img1,(int(pt1[0,0]),int(pt1[0,1])),5,line_color,-1) 
        img2 = cv2.circle(img2,(int(pt2[0,0]),int(pt2[0,1])),5,line_color,-1) 
    return img1, img2 

# Drawing the epipolar lines for a pair images given a set of matched features
def draw_epipolar_lines(rgb_img1,rgb_img2,left_points,right_points):
    # Creating deepcopies of each image to prevent unintended annotations
    img1 = copy.deepcopy(cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2GRAY))
    img2 = copy.deepcopy(cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2GRAY))
    # Calculating the fundamental matrix using RANSAC to get the association between features
    # This may be a repeated step in the overall workflow considering that F has already been computed    
    F, mask = cv2.findFundamentalMat(left_points,right_points,cv2.FM_RANSAC,0.1,0.99)
    left_points = left_points[mask.ravel() == 1] 
    right_points = right_points[mask.ravel() == 1]
    # Calculating the lines to be drawn on each image
    lines_left = cv2.computeCorrespondEpilines(right_points.reshape(-1,1,2),2,F).reshape(-1,3)
    lines_right = cv2.computeCorrespondEpilines(left_points.reshape(-1,1,2),1,F) .reshape(-1,3)
    # Drawing and displaying the epipolar lines on each image  
    img1_epipolar_lines, _ = draw_lines_on_img(img1,img2,left_points,right_points,lines_left) 
    img2_epipolar_lines, _ = draw_lines_on_img(img1,img2,right_points,left_points,lines_right) 
    plt.subplot(121), plt.imshow(img1_epipolar_lines) 
    plt.subplot(122), plt.imshow(img2_epipolar_lines) 
    plt.show()

 # Obtaining the Fundamental and Essential Matrix along with the rotational and translational transformation between the two unrectified images
def unrectified_image_association(left_img_points,right_img_points,K1,K2):
    # Getting the Fundamental Matrix using RANSAC
    fundamental_matrix, mask = cv2.findFundamentalMat(left_img_points,right_img_points,cv2.FM_RANSAC,0.1,0.99)
    # Getting the Essential Matrix by using the formula E = (K1.T)F(K2)
    essential_matrix = np.dot(K1.T,np.dot(fundamental_matrix,K2))
    # Using Singular-Value Decomposition (SVD) to get the rotational and translational transformations (R and T respectively) between the two images 
    U, S, V = np.linalg.svd(essential_matrix)
    #W = np.array([[0,-1,0],[1,0,0],[0,0,1]]).astype(np.float32)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    R = np.dot(U,np.dot(W,V))
    T = U[:,2]
    # Displaying the Fundamental Matrix pre-rectification
    print("The Fundamental Matrix pre-rectification is: ")
    print(fundamental_matrix)
    # Displaying the Essential Matrix pre-rectification
    print("The Essential Matrix pre-rectification is: ")
    print(essential_matrix)
    # Displaying the Rotational Transformation Matrix pre-rectification
    print("The Rotation Matrix computed from the Essential Matrix pre-rectification is: ")
    print(R)
    # Displaying the Translational Transformation Matrix pre-rectification
    print("The Translation Vector computed from the essential Matrix pre-rectification is: ")
    print(T)
    return fundamental_matrix, essential_matrix, R, T

# Rectifying the stereo-image pair
def rectify_stereo_pair(img1,img2,fundamental_matrix,left_img_points,right_img_points):
    # Ontaining the dimensions of both images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # Getting the homographies needed to be applied on each image to rectify them
    # Can use cv2.stereoRectify as well, but using cv2.stereoRectifyUncalibrated to directly get the homography transformations as an output
    _, H1, H2 = cv2.stereoRectifyUncalibrated(left_img_points,right_img_points,fundamental_matrix,imgSize=[w1,h1])
    # Calculating the new fundamental matrix post-rectification
    F_new = np.dot(np.linalg.inv(H2),np.dot(fundamental_matrix,np.linalg.inv(H1)))
    # Rectifying the the image pair by warping using the homography transformation
    img1_rect = cv2.warpPerspective(copy.deepcopy(img1),H1,[w1,h1])
    img2_rect = cv2.warpPerspective(copy.deepcopy(img2),H2,[w2,h2])
    # Generating a concatenated version of the two images next to eachother to visually ensure rectification
    img_concat = np.concatenate([copy.deepcopy(img1_rect),copy.deepcopy(img2_rect)],axis=1)
    img_concat = cv2.cvtColor(copy.deepcopy(img_concat),cv2.COLOR_BGR2RGB)
    # Displaying the homography matrix to rectify the first/left image
    print("The homography matrix (H1) for rectifying the first/left image: ")
    print(H1)
    # Displaying the homography matrix to rectify the second/right image
    print("The homography matrix (H2) for rectifying the second/right image: ")
    print(H2)
    # Displaying the new fundamental matrix for the rectified stereo pair
    print("The new Fundamental_matrix post-rectification is: ")
    print(F_new)
    # Displaying the two rectfied images next to eachother (only horizontal translation should be observed)
    plt.imshow(img_concat)
    plt.show()
    return img1_rect, img2_rect, H1, H2

# Generating a disparity map from the rectified image pair using the block-matching algorithm
def generate_disparity_map(img1_rect,img2_rect,window_size=11):
    # Downsampling and grayscaling the images for block-matching to enable faster execution
    imgL = cv2.resize(copy.deepcopy(img1_rect),(int(img1_rect.shape[1]/4),int(img1_rect.shape[0]/4)))
    imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY).astype(np.int32)
    imgR = cv2.resize(copy.deepcopy(img2_rect),(int(img2_rect.shape[1]/4),int(img2_rect.shape[0]/4)))
    imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY).astype(np.int32)
    # Initialzing the block-matching algorithm parameters
    h, w = imgL.shape
    disparity_map = np.zeros((h,w))
    disp_block_width = w - 2*window_size
    print("Generating the Disparity Map. This may take a couple seconds.")
    # Looping through window locations across the height of the left image
    for i in range(window_size,h-window_size):
        imgL_blocks = []
        imgR_blocks = []
        # Looping through the window locations across the width of the left image
        for j in range(window_size,w-window_size):
            left_img_block = imgL[i:i+window_size,j:j+window_size]
            right_img_block = imgR[i:i+window_size,j:j+window_size]
            imgL_blocks.append(left_img_block.flatten())
            imgR_blocks.append(right_img_block.flatten())
        # Generating and adding the block corresponding to the current window location
        imgL_blocks = np.array(imgL_blocks)
        imgR_blocks = np.array(imgR_blocks)
        imgL_blocks = np.repeat(imgL_blocks[:,:,np.newaxis],disp_block_width,axis=2)
        imgR_blocks = np.repeat(imgR_blocks[:,:,np.newaxis],disp_block_width,axis=2).T
        # Using the sum of absolute differences (SAD) method to find the best matching block in the right image with the left image
        blocks_diff = np.sum(np.abs(imgL_blocks-imgR_blocks),axis=1)
        ideal_idx = np.argmin(blocks_diff,axis=0)
        disparity_block = np.abs(ideal_idx - np.linspace(0,disp_block_width,disp_block_width,dtype=int)).reshape(1,disp_block_width)
        # Using the best matching block to get the disparity values for that row across the height dimension
        disparity_map[i,0:disp_block_width] = disparity_block 
    # Scaling the disparity map to 8-bit [0-255] pixel values
    disparity_map = ((disparity_map/disparity_map.max())*255).astype(np.int32)
    # Displaying the disparity map is a heat-map
    print("The Disparity Map using a Heatmap Visualization: ")
    plt.imshow(disparity_map,cmap='hot',interpolation='nearest')
    plt.show()
    # Displaying the disparity map is a grayscale image
    print("The Disparity Map using a Grayscale Visualization: ")
    plt.imshow(disparity_map,cmap='gray',interpolation='nearest')
    plt.show()
    return disparity_map

# Generating a depth map of the scene using the disparity map
def generate_depth_map(disparity_map,baseline,focal_length):
    # Using the formula depth = (focal_length*baseline)/disparity for eacxh pixel of the disparity map
    depth_map = (baseline*focal_length)/(disparity_map+np.exp(-7))
    # Thresholding and scaling the depth map to 8-bit [0-255] pixel values
    print(depth_map.max())
    print(depth_map.min())
    depth_map[depth_map>1000000] = 1000000
    depth_map = ((depth_map/depth_map.max())*255).astype(np.int32)
    # Displaying the depth map is a heat-map
    print("The Depth Map using a Heatmap Visualization: ")
    plt.imshow(depth_map,cmap='hot',interpolation='nearest')
    plt.show()
    # Displaying the depth map as a grayscale image
    print("The Depth Map using a Grayscale Visualization: ")
    plt.imshow(depth_map,cmap='gray',interpolation='nearest')
    plt.show()
    return depth_map

# A wrapper function to sequentially detect features, compute matrices, rectify the stereo pair, and compute disparity and depth maps
def stereo_vision_wrapper(img1,img2,K1,K2,baseline):
    # Making deepcopies of the 
    rgb_img1 = copy.deepcopy(img1)
    rgb_img2 = copy.deepcopy(img2)
    # Ectracting the Focal Length of the camera from the camera intrinsic matrix.The line below is valid only when K1 == K2 and fx == fy
    focal_length = K1[0,0]
    # Detecting SIFT Featutre Keypoints and matching them using a FLANN Matcher
    kp1, features1 = detect_sift_features(copy.deepcopy(rgb_img1))
    kp2, features2 = detect_sift_features(copy.deepcopy(rgb_img2))
    good_matches = FLANN_match_features(features1,features2)
    # Extracting just the SIFT feature keypoint locations in each image
    left_img_points = np.array([kp1[matches.queryIdx].pt for matches in good_matches]).reshape(-1,1,2).astype(np.float32)
    right_img_points = np.array([kp2[matches.trainIdx].pt for matches in good_matches]).reshape(-1,1,2).astype(np.float32)
    # Obtaining the Fundamental and Essential Matrix along with the Rotation and Translation transform for the unrectified image pair
    F, E, R, T = unrectified_image_association(left_img_points,right_img_points,K1,K2)
    # Drawing the epipolar lines (non-horizontal) for the unrectified image pair
    draw_epipolar_lines(copy.deepcopy(rgb_img1),copy.deepcopy(rgb_img2),left_img_points,right_img_points)
    # Rectifying the stereo image pair
    img1_rect, img2_rect, H1, H2 = rectify_stereo_pair(rgb_img1,rgb_img2,F,left_img_points,right_img_points)
    # Appropriately transforming the SIFT Feature Keypoint locations post-rectification
    left_img_points = cv2.perspectiveTransform(left_img_points,H1).reshape(-1,1,2)
    right_img_points = cv2.perspectiveTransform(right_img_points,H2).reshape(-1,1,2)
    # Re-drawring the epipolar lines (now horizontal) after rectification
    draw_epipolar_lines(copy.deepcopy(img1_rect),copy.deepcopy(img2_rect),left_img_points,right_img_points)
    # Generating a Disparity Map for the rectified stereo-image pair
    disparity_map = generate_disparity_map(img1_rect,img2_rect)
    # Generating a Depth Map for the rectified stereo-image pair
    depth_map = generate_depth_map(disparity_map,baseline,focal_length)

# Performing stereo vision analysis for each image set
def main():
    global datasets
    for dataset in datasets:
        img1 = cv2.imread(dataset["img1_path"])
        img2 = cv2.imread(dataset["img2_path"])
        K1 = dataset["K1"]
        K2 = dataset["K2"]
        baseline = dataset["baseline"]
        stereo_vision_wrapper(img1,img2,K1,K2,baseline)

if __name__ == "__main__":
    main()
