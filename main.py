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

def draw_epipolar_lines(rgb_img1,rgb_img2,left_points,right_points):
    img1 = copy.deepcopy(cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2GRAY))
    img2 = copy.deepcopy(cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2GRAY))    
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
    print("The Fundamental Matrix pre-rectification is: ")
    print(fundamental_matrix)
    print("The Essential Matrix pre-rectification is: ")
    print(essential_matrix)
    print("The Rotation Matrix computed from the Essential Matrix pre-rectification is: ")
    print(R)
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
    img_concat = np.concatenate([copy.deepcopy(img1_rect),copy.deepcopy(img2_rect)],axis=1)
    print("The homography matrix (H1) for rectifying the first/left image: ")
    print(H1)
    print("The homography matrix (H2) for rectifying the second/right image: ")
    print(H2)
    print("The new Fundamental_matrix post-rectification is: ")
    print(F_new)
    img_concat = cv2.cvtColor(copy.deepcopy(img_concat),cv2.COLOR_BGR2RGB)
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
    disparity_map = ((disparity_map*255/disparity_map.max())).astype(np.int32)
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
    depth_map = (baseline*focal_length)/(disparity_map+np.exp(-10))
    # Thresholding and scaling the depth map to 8-bit [0-255] pixel values
    depth_map[depth_map>100000] = 100000
    depth_map = ((depth_map*255/depth_map.max())).astype(np.int32)
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
    F, E, R, T = unrectified_image_association(good_matches,kp1,kp2,K1,K2)
    # Drawing the epipolar lines (non-horizontal) for the unrectified image pair
    draw_epipolar_lines(rgb_img1,rgb_img2,left_img_points,right_img_points)
    # Rectifying the stereo image pair
    img1_rect, img2_rect, H1, H2 = rectify_stereo_pair(rgb_img1,rgb_img2,F,left_img_points,right_img_points)
    # Appropriately transforming the SIFT Feature Keypoint locations post-rectification
    left_img_points = cv2.perspectiveTransform(left_img_points, H1).reshape(-1,1,2)
    right_img_points = cv2.perspectiveTransform(right_img_points, H2).reshape(-1,1,2)
    # Re-drawring the epipolar lines (now horizontal) after rectification
    draw_epipolar_lines(img1_rect,img2_rect,left_img_points,right_img_points)
    # Generating a Disparity Map for the rectified stereo-image pair
    disparity_map = generate_disparity_map(img1_rect,img2_rect)
    # Generating a Depth Map for the rectified stereo-image pair
    depth_map = generate_depth_map(disparity_map,baseline,focal_length)

def main():
    img1 = cv2.imread("data/storageroom/im1.png")
    img2 = cv2.imread("data/storageroom/im0.png")
    #K1 = np.array([[1746.24,0,14.88],[0,1746.24,534.11],[0,0,1]])
    #K2 = np.array([[1746.24,0,14.88],[0,1746.24,534.11],[0,0,1]])
    K1 = np.array([[1742.11,0,804.90],[0,1742.11,541.22],[0,0,1]])
    K2 = np.array([[1742.11,0,804.90],[0,1742.11,541.22],[0,0,1]])
    #d1 = np.array([0,0,0,0]).astype(np.float32)
    #d2 = np.array([0,0,0,0]).astype(np.float32)
    baseline = 221.76
    stereo_vision_wrapper(img1,img2,K1,K2,baseline)

if __name__ == "__main__":
    main()


























