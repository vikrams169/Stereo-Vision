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

# Calculating the homography between two images using 8 sets of corresponding points (from matched features)
# Using Singular Value Decomposition (SVD) for the approximate homography matrix calculation
def calculate_homography(img1_sample,img2_sample):
    A = [] # Initiliazing the A matrix
    # For each matched feature, adding the corresponding rows to the A matrix
    for i in range(img1_sample.shape[0]):
        x1, y1 = img1_sample[i][0][0], img1_sample[i][0][1]
        x2, y2 = img2_sample[i][0][0], img2_sample[i][0][1]
        A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
        A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2])
    # Using SVD to approximate the homography from the A matrix
    A = np.array(A).astype(np.float32)
    U, S, V = np.linalg.svd(A,full_matrices=True)
    homography = V[8,:].reshape((3, 3))
    return homography

# Calculating the fundamental matrix between two images using 8 sets of corresponding points (from matched features)
# Using Singular Value Decomposition (SVD) for the approximate fundamental matrix calculation
def calculate_fundamental_matrix(img1_sample,img2_sample):
    A = [] # Initiliazing the A matrix
    # For each matched feature, adding the corresponding rows to the A matrix
    for i in range(img1_sample.shape[0]):
        x1, y1 = img1_sample[i][0][0], img1_sample[i][0][1]
        x2, y2 = img2_sample[i][0][0], img2_sample[i][0][1]
        A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    # Using SVD to approximate the homography from the fundamental matrix
    A = np.array(A).astype(np.float32)
    U, S, V = np.linalg.svd(A,full_matrices=True)
    fundamental_matrix = V[8,:].reshape((3, 3))
    return fundamental_matrix

# Finding the transformed points using a given homography
def apply_perspective_transform(img_points,homography):
    img_points_transformed = [] # Initializing the final set of transformed points
    # Applying a perspective transformation by matrix multiplying each point with the homography matrix
    for i in range(len(img_points)):
        point = np.array([[img_points[i][0][0]],[img_points[i][0][1]],[1]])
        transformed_point = np.dot(homography,point)
        img_points_transformed.append([transformed_point[0]/(transformed_point[2]+0.00001),transformed_point[1]/(transformed_point[2]+0.00001)])
    img_points_transformed = np.array(img_points_transformed).reshape(-1,1,2).astype(np.int32) # Reshaping the array of transformed points to the appropriate dimension
    return img_points_transformed

# Using RANSAC to find the best fundamnetal matrix approximation from the current set of good matches
def ransac(good_matches,k1,k2,num_iterations=5000,inlier_threshold=5):
    # Obtaining the set of pixel locations of the features in each of the images
    img1_points = np.array([k1[matches.queryIdx].pt for matches in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    img2_points = np.array([k2[matches.trainIdx].pt for matches in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    # Initializing the best homography and indices and number of maximum inliers from the matches
    best_fundamental_matrix = None
    max_inliers = 0
    max_inlier_indices = []
    # Iteratively trying to find the best homography in each RANSAC iteration
    for it in range(num_iterations):
        # Choosing a random sample of 8 matches from all good matches
        random_indices = np.random.choice(img1_points.shape[0],8,replace=False)
        img1_sample = img1_points[random_indices]
        img2_sample = img2_points[random_indices]
        # Calculating the current homography between the matches in the chosen sample
        current_homography = calculate_homography(img1_sample,img2_sample)
        # Transforming the set of keypoints in the first image to the perspective of the second image using the computed homography
        img1_points_transformed = apply_perspective_transform(img1_points,current_homography)
        # Calculating the indices and the number of inliers for the current chosen keypoints sample and computed homography
        dist = np.linalg.norm(img1_points_transformed-img2_points,axis=-1)
        num_inliers = np.sum(dist<inlier_threshold)
        inlier_indices = list(np.array(dist<inlier_threshold).ravel())
        # Updating the best fundamental matrix, and the indices and number of maximum inliers if the current sample has the most inliers among all RANSAC iterations so far
        if num_inliers > max_inliers:
            best_fundamental_matrix = calculate_fundamental_matrix(img1_sample,img2_sample)
            max_inliers = num_inliers
            max_inlier_indices = inlier_indices
    return best_fundamental_matrix, max_inlier_indices

def calculate_essential_matrix(fundamental_matrix,K1,K2):
    essential_matrix = np.dot(K1.T,np.dot(fundamental_matrix,K2))
    return essential_matrix

def disparity_map(img1_rect,img2_rect,Q,baseline=678.37,f=1746.24):
    imgL = cv2.cvtColor(img1_rect,cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rect,cv2.COLOR_BGR2GRAY)
    stereo_function = cv2.StereoBM.create(numDisparities=64,blockSize=15)
    disparity = stereo_function.compute(imgL,imgR)
    plt.imshow(disparity,cmap='gray')
    plt.show()
    '''depth = cv2.reprojectImageTo3D(disparity,Q)[:,:,2]
    depth = (depth-depth.min()/depth.max()-depth.min())*255
    depth = depth.astype(np.int32)
    print(depth.shape)'''
    depth = np.zeros((disparity.shape))
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            depth[i,j] = (baseline*f)/disparity[i,j]
    plt.imshow(depth,cmap='hot')
    plt.show()

def compute_matrices(img1,img2,good_matches,kp1,kp2,K1,K2):
    #print(img1.shape)
    #print(img2.shape)
    img1_points = np.array([kp1[matches.queryIdx].pt for matches in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    img2_points = np.array([kp2[matches.trainIdx].pt for matches in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    fundamental_matrix, mask = cv2.findFundamentalMat(img1_points,img2_points,cv2.FM_RANSAC,0.1,0.99)
    essential_matrix = np.dot(K1.T,np.dot(fundamental_matrix,K2))
    U, S, V = np.linalg.svd(essential_matrix)
    #W = np.array([[0,-1,0],[1,0,0],[0,0,1]]).astype(np.float32)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    R = np.dot(U,np.dot(W,V))
    T = U[:,2]
    d1, d2 = np.array([0,0,0,0]), np.array([0,0,0,0])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, d1, K2, d2, [img1.shape[1],img1.shape[0]], R, T, alpha=1.0)
    mapx1, mapy1 = cv2.initUndistortRectifyMap(K1, d1, R1, K1, [img1.shape[1],img1.shape[0]], cv2.CV_32F)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(K2, d2, R2, K2, [img1.shape[1],img1.shape[0]], cv2.CV_32F)
    img_rect1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)
    img_rect2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR)
    #print(img_rect1.shape)
    #print(img_rect2.shape)
    img_concat = np.concatenate([copy.deepcopy(img_rect1),copy.deepcopy(img_rect2)],axis=1)
    img_concat =cv2.cvtColor(img_concat,cv2.COLOR_BGR2RGB)
    plt.imshow(img_concat)
    plt.show()
    disparity_map(img_rect1,img_rect2,Q)
    return fundamental_matrix, essential_matrix, R, T

# A wrapper function that directly outputs the best fundamental matrix and required visualizations from two input RGB images
def fundamental_and_essential_matrix_wrapper(img1,img2,K1,K2):
    # Making deep copies of each image to avoid writing visualization features to the original images
    rgb_img1 = copy.deepcopy(img1)
    rgb_img2 = copy.deepcopy(img2)
    # Detecting the SIFT features for each image
    k1, d1 = detect_sift_features(copy.deepcopy(rgb_img1))
    k2, d2 = detect_sift_features(copy.deepcopy(rgb_img2))
    # Detecting the FLANN matches from the SIFT features between the two images
    good_matches = FLANN_match_features(d1,d2)
    # Drawing the FLANN matches for visualization
    #draw_match12_flann = cv2.drawMatches(copy.deepcopy(rgb_img1),k1,copy.deepcopy(rgb_img2),k2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Using RANSAC to get the best fundamental matrix between the two images
    #best_fundamental_matrix, inlier_indices = ransac(good_matches,k1,k2)
    #best_fundamental_matrix, mask = compute_matrices(good_matches,k1,k2,K1,K2)
    # Using the inlier indices from RANSAC, getting the exact inlier matches from the earlier obtained FLANN matches
    #new_good_matches = []
    #for i in range(len(inlier_indices)):
        #if int(inlier_indices[i]) == 1:
            #new_good_matches.append(good_matches[i])
    # Drawing the RANSAC macthes for visualization
    #draw_match12_ransac = cv2.drawMatches(copy.deepcopy(rgb_img1),k1,copy.deepcopy(rgb_img2),k2,new_good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    best_fundamental_matrix, best_essential_matrix, R, T = compute_matrices(img1,img2,good_matches,k1,k2,K1,K2)
    return best_fundamental_matrix, best_essential_matrix, R, T#, copy.deepcopy(kp_img1), copy.deepcopy(kp_img2), copy.deepcopy(draw_match12_flann), copy.deepcopy(draw_match12_ransac)

def main():
    img1 = cv2.imread("data/classroom/im0.png")
    img2 = cv2.imread("data/classroom/im1.png")
    K1 = np.array([[1746.24,0,14.88],[0,1746.24,534.11],[0,0,1]])
    K2 = np.array([[1746.24,0,14.88],[0,1746.24,534.11],[0,0,1]])
    baseline = 678.37
    F, E, R, T = fundamental_and_essential_matrix_wrapper(img1,img2,K1,K2)
    print("Fundamental Matrix: ")
    print(F)
    print("***************************************************************")
    print("Essential Matrix: ")
    print(E)
    print("***************************************************************")
    print("Rotation Matrix: ")
    print(R)
    print("***************************************************************")
    print("Translation Matrix: ")
    print(T)

if __name__ == "__main__":
    main()


























