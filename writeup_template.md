## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[img_find_chessboard_corner]: ./test_images_output/find_chessboard_corner.png
[img_calibrate_undistort]: ./test_images_output/calibrate_undistort.png 
[img_unwarp]: ./test_images_output/unwarp.png
[img_color-rgb]: ./test_images_output/color-rgb.png
[img_color-hsv]: ./test_images_output/color-hsv.png
[img_color-lab]: ./test_images_output/color-lab.png

[img_abs_sobel_thresh]: ./test_images_output/abs_sobel_thresh.png
[img_mag_thresh]: ./test_images_output/mag_thresh.png
[img_dir_threshold]: ./test_images_output/dir_threshold.png
[img_mag_thresh_and_dir_threshold]: ./test_images_output/mag_thresh_and_dir_threshold.png
[img_hls_sthresh]: ./test_images_output/hls_sthresh.png
[img_hls_lthresh]: ./test_images_output/hls_lthresh.png
[img_lab_bthresh]: ./test_images_output/lab_bthresh.png
[img_pipeline]: ./test_images_output/pipeline.png
[img_find_histogram]: ./test_images_output/find_histogram.png
[img_fit_polynomial]: ./test_images_output/fit_polynomial.png
[img_search_around_poly]: ./test_images_output/search_around_poly.png
[img_draw_lane]: ./test_images_output/draw_lane.png
[img_draw_data]: ./test_images_output/draw_data.png

[video_project_video_output]: ./project_video_output.mp4
[video_challenge_video_output]: ./challenge_video_output.mp4
[video_harder_challenge_video_output]: ./harder_challenge_video_output.mp4


---
**File Location**

* Notebook: examples/example.ipynb
* Test images output: test_images_output/*
* Project video output: project_video_output.mp4
* Challenge video output: challenge_video_output.mp4
* Harder challenge video output: harder_challenge_video_output.mp4


---
## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


---

## 1. First, I'll compute the camera calibration using chessboard images

Use funtinon:
1. find_chessboard_corner: Find out the cheessboard corners and use chessboard images to obtain image points and object points

```
def find_chessboard_corner(img):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        return img
    else:
        return None
```
### Run find_chessboard_corner on calibration images (input path: ../camera_cal/):

![alt text][img_find_chessboard_corner]

---

## 2. Next, I'll corret the image distortion

Funtinon:
1. calibrate_undistort: Use the OpenCV functions cv2.calibrateCamera() and cv2.undistort() to compute the calibration and undistortion.

```
def calibrate_undistort(img):
    # Use cv2.calibrateCamera() and cv2.undistort()
    # undist = np.copy(img)  # Delete this line
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    return dst
```
### Run calibrate_undistort on test images (input path: ../test_images/):

![alt text][img_calibrate_undistort]

---

## 3. Next, I'll apply a perspective transform

Funtinon:
1. unwarp: To transofrm an image to bird’s-eye view that let’s us view a lane from above
  * Define 4 source points 
  * Define 4 destination points 
  * Use cv2.getPerspectiveTransform() to get M, the transform matrix
  * use cv2.warpPerspective() to apply M and warp your image to a top-down view

```
def unwarp(img):
    img_size = (img.shape[1], img.shape[0])
        
    # define source and destination points for transform
    src = np.float32([(575,464),
                      (707,464), 
                      (258,682), 
                      (1049,682)])
    dst = np.float32([(450,0),
                      (img_size[0]-450,0),
                      (450,img_size[1]),
                      (img_size[0]-450,img_size[1])])
    
    M = cv2.getPerspectiveTransform(src, dst)
    
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped
```
### Run unwarp on test images (input path: ../test_images/):

![alt text][img_unwarp]

---

## 4. Next, I'll test on different color space channels (RGB, HLS, LAB)

### (1) Display RGB color on test images (input path: ../test_images/):

![alt text][img_color-rgb]


### (2) Display HSV color on test images (input path: ../test_images/):

![alt text][img_color-hsv]


### (3) Display LAB color on test images (input path: ../test_images/):

![alt text][img_color-lab]


---


## 5. Next, I'll test a sobel absolute threshold

Funtinon:
1. abs_sobel_thresh: To take an absolute value and applies a threshold for Sobel x or y
  * Convert to grayscale
  * Take the derivative in x or y given orient = 'x' or 'y'
  * Take the absolute value of the derivative or gradient
  * Scale to 8-bit (0 - 255) then convert to type = np.uint8
  * Create a mask of 1's where the scaled gradient magnitude, # is > thresh_min and < thresh_max
  * Return this mask as your binary_output image

```
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output
```

### Run abs_sobel_thresh on test images (input path: ../test_images/):

![alt text][img_abs_sobel_thresh]

---

## 6. Next, I'll test a sobel magnitude threshold

Funtinon:
1. mag_thresh: To compute the magnitude of the gradient and applies a threshold for Sobel x or y
  * Convert to grayscale
  * Take the derivative in x or y given orient = 'x' or 'y'
  * Calculate the magnitude 
  * Scale to 8-bit (0 - 255) then convert to type = np.uint8
  * Create a mask of 1's where the scaled gradient magnitude, # is > thresh_min and < thresh_max
  * Return this mask as your binary_output image

```
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
```

### Run mag_thresh on test images (input path: ../test_images/):

![alt text][img_mag_thresh]


---

## 7. Next, I'll test a sobel direction threshold

Funtinon:
1. dir_threshold: To compute the direction of the gradient and applies a threshold for Sobel x or y
  * Convert to grayscale
  * Take the derivative in x or y given orient = 'x' or 'y'
  * Take the absolute value of the derivative or gradient
  * Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
  * Create a binary mask where direction thresholds are met
  * Return this mask as your binary_output image

```
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
```

### Run dir_threshold on test images (input path: ../test_images/):

![alt text][img_dir_threshold]

### Run mag_thresh and dir_threshold on test images (input path: ../test_images/):

![alt text][img_mag_thresh_and_dir_threshold]

---

## 8. Next, I'll test a HLS S-Channel threshold

Funtinon:
1. hls_sthresh: To threshold the S-channel of HLS
  * Convert to HLS color space
  * Apply a threshold to the S channel
  * Return a binary image of threshold result

```
def hls_sthresh(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(hls[:,:,2])
    binary_output[(hls[:,:,2] > thresh[0]) & (hls[:,:,2] <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output
```

### Run hls_sthresh on test images (input path: ../test_images/):

![alt text][hls_sthresh]

---

## 9. Next, I'll test a HLS L-Channel threshold

Funtinon:
1. hls_lthresh: To threshold the L-channel of HLS
  * Convert to HLS color space
  * Apply a threshold to the L channel
  * Return a binary image of threshold result

```
def hls_lthresh(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output
```

### Run hls_lthresh on test images (input path: ../test_images/):

![alt text][img_hls_lthresh]


---

## 10. Next, I'll test a LAB B-Channel threshold

Funtinon:
1. lab_bthresh: To threshold the B-channel of LAB
  * Convert to LAB color space
  * Apply a threshold to the B channel
  * Return a binary image of threshold result

```
def lab_bthresh(img, thresh=(0, 255)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the B channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output
```

### Run lab_bthresh on test images (input path: ../test_images/):

![alt text][img_lab_bthresh]


---

## 11. Next, I'll creat a image processing pipeline

Funtinon:
1. pipeline: To combine all processing function above, reads raw image and returns binary image with lane lines identified
  * Undistort
  * Perspective Transform
  * (Sobel Absolute)
  * (Sobel Magnitude)
  * (Sobel Direction)
  * (HLS S-channel Threshold)
  * HLS L-channel Threshold
  * LAB B-channel Threshold

```
def pipeline(img):
    # Undistort
    img_undistort = calibrate_undistort(img)
    
    # Perspective Transform
    img_unwarp = unwarp(img_undistort)

    # Sobel Absolute
    #img_sobelAbs = abs_sobel_thresh(img_undistort, orient='x', thresh_min=20, thresh_max=100)

    # Sobel Magnitude 
    #img_sobelMag = mag_thresh(img_undistort, sobel_kernel=25, mag_thresh=(25, 255))
    
    # Sobel Direction
    #img_sobelDir = dir_threshold(img_undistort, sobel_kernel=7, thresh=(0, 0.09))
    
    # HLS S-channel Threshold
    #img_SThresh = hls_sthresh(img_unwarp, thresh=(125, 255))

    # HLS L-channel Threshold
    img_LThresh = hls_lthresh(img_unwarp, thresh=(220, 255))

    # Lab B-channel Threshold
    img_BThresh = lab_bthresh(img_unwarp, thresh=(190,255))
    
    # Combine HLS L-channel and LAB B-channel channel thresholds
    combined = np.zeros_like(img_BThresh)
    combined[(img_LThresh == 1) | (img_BThresh == 1)] = 1
    
    # Combine All
#     combined[(img_sobelAbs == 1) | (img_sobelMag == 1) | (img_sobelDir == 1) | (img_SThresh == 1) | (img_LThresh == 1) | (img_BThresh == 1)] = 1
    
    return combined
```

### Run pipeline on test images (input path: ../test_images/):

![alt text][img_pipeline]


---

## 12. Next, I'll use histogram peaks to find the line

Funtinon:
1. find_histogram: To create a histogram and find find the line

```
def find_histogram(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram
```

### Run find_histogram on test images (input path: ../test_images/):

![alt text][img_find_histogram]


---

## 13. Next, I'll create sliding windows and fit a polynomial

Funtinon:
1. find_lane_pixels: To find the window boundaries and all pixels within those boundaries. If there are more than minpix, slide the window over to the mean of these pixels.
2. fit_polynomial: To fit a polynomial to the line by using sliding window

```
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Rectangle data for visualization
    rectangle_data = []

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        
        # Draw the windows on the visualization image
#         cv2.rectangle(out_img,(win_xleft_low,win_y_low),
#         (win_xleft_high,win_y_high),(0,255,0), 2) 
#         cv2.rectangle(out_img,(win_xright_low,win_y_low),
#         (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    visualization_data = (rectangle_data, histogram)

#     return leftx, lefty, rightx, righty, visualization_data
    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data


def fit_polynomial(binary_warped):
    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = find_lane_pixels(binary_warped)

    h = binary_warped.shape[0]
    left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
    right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
#     print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)

    rectangles = visualization_data[0]
    histogram = visualization_data[1]

    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((binary_warped, binary_warped, binary_warped))*255)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    for rect in rectangles:
    # Draw the windows on the visualization image
        cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2) 
        cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2) 
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]
#     plt.imshow(out_img)
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
#     plt.xlim(0, 1280)
#     plt.ylim(720, 0)
    
#     return out_img
    return out_img, left_fitx, right_fitx, ploty

```

### Run fit_polynomial on test images (input path: ../test_images/):

![alt text][img_fit_polynomial]


---

## 14. Next, I'll use the previous polynomial to skip the sliding window

Funtinon:
1. fit_poly: To fit a polynomial to all the relevant pixels we've found in sliding windows
2. search_around_poly: To set the area to search for activated pixels based on margin out from the fit polynomial

```
def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Get left_fit, right_fit, left_lane_inds, right_lane_inds
    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = find_lane_pixels(binary_warped)

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
#     left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
#                     left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
#                     left_fit[1]*nonzeroy + left_fit[2] + margin)))
#     right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
#                     right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
#                     right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
#     return result
    return result, left_fitx, right_fitx, ploty
```

### Run search_around_poly on test images (input path: ../test_images/):

![alt text][img_search_around_poly]


---

## 15. Next, I'll calculate the radius of curvature and distance from lane center

Funtinon:
1. calc_curv_rad_and_center_dist: To determine radius of curvature and distance from lane center

```
# Method to determine radius of curvature and distance from lane center 
# based on binary image, polynomial fit, and L and R lane pixel indices
def calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = bin_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)
  
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds] 
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]
    
    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
    
    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts 
    if r_fit is not None and l_fit is not None:
        car_position = bin_img.shape[1]/2
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    return left_curverad, right_curverad, center_dist
```

### Run calc_curv_rad_and_center_dist on test images (input path: ../test_images/):

```
Radius of curvature for straight_lines1.jpg : 1777.83777154 m, 9647.0878312 m
Distance from lane center for straight_lines1.jpg : 0.00183729206013 m
Radius of curvature for test2.jpg : 448.263413652 m, 1211.11186698 m
Distance from lane center for test2.jpg : -0.391118236481 m
Radius of curvature for straight_lines2.jpg : 2312.12271075 m, 10012.302298 m
Distance from lane center for straight_lines2.jpg : -0.0253793218877 m
Radius of curvature for test4.jpg : 883.232745182 m, 409.158854314 m
Distance from lane center for test4.jpg : -0.313903404806 m
Radius of curvature for test1.jpg : 489.711733398 m, 87.5903118092 m
Distance from lane center for test1.jpg : -1.71403417275 m
Radius of curvature for test6.jpg : 898.022334635 m, 556.31061785 m
Distance from lane center for test6.jpg : -0.278246123389 m
Radius of curvature for test5.jpg : 377.758027697 m, 1629.2054546 m
Distance from lane center for test5.jpg : 0.00566442052808 m
Radius of curvature for test3.jpg : 1230.45676448 m, 534.655607459 m
Distance from lane center for test3.jpg : -0.134716710015 m
```

---

## 16. Next, I'll draw the detected lane back onto the original image

Funtinon:
1. draw_lane: To draw the detected lane back onto the original image

```
def draw_lane(original_img, binary_img):
    new_img = np.copy(original_img)
#     if l_fit is None or r_fit is None:
#         return original_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    
    # Get left_fit, right_fit
    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = find_lane_pixels(binary_img)

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)
    
    # Get Minv
    img_size = (binary_img.shape[1], binary_img.shape[0])
    src = np.float32([(575,464),
                      (707,464), 
                      (258,682), 
                      (1049,682)])
    dst = np.float32([(450,0),
                      (img_size[0]-450,0),
                      (450,img_size[1]),
                      (img_size[0]-450,img_size[1])])    
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result
```

### Run draw_lane on test images (input path: ../test_images/):

![alt text][img_draw_lane]


---

## 17. Next, I'll draw the curvature radius and distance from center data onto the original image

Funtinon:
1. draw_data: To draw the curvature radius and distance from center data onto the original image

```
def draw_data(original_img, binary_img):
    new_img = np.copy(original_img)
    h = new_img.shape[0]
    
    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = find_lane_pixels(binary_img)
    rad_l, rad_r, d_center = calc_curv_rad_and_center_dist(binary_img, left_fit, right_fit, left_lane_inds, right_lane_inds)
    
#     print('Radius of curvature for', name, ':', rad_l, 'm,', rad_r, 'm')
#     print('Distance from lane center for', name, ':', d_center, 'm')
    curv_rad = (rad_l+rad_r)/2
    center_dist = d_center

    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return new_img
```

### Run draw_data on test images (input path: ../test_images/):

![alt text][img_draw_data]


---

## 19. Next, I'll define a line class to keep track the characteristics of each line detection

Class:
1. Line: To keep track of all the interesting parameters you measure from frame to frame

```
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #number of detected pixels
        self.px_count = None
    def add_fit(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            if (self.diffs[0] > 0.001 or \
               self.diffs[1] > 1.0 or \
               self.diffs[2] > 100.) and \
               len(self.current_fit) > 0:
                # bad fit! abort! abort! ... well, unless there are no fits in the current_fit queue, then we'll take it
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)

```
---
## 20. Next, I'll define a complete image processing pipeline

Class:
1. process_image: To complete image processing pipeline by existing processing function

```
def process_image(img):
    new_img = np.copy(img)
    binary_img = pipeline(new_img)
    
    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = find_lane_pixels(binary_img)
    
    # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
#     if not l_line.detected or not r_line.detected:
#         l_fit, r_fit, l_lane_inds, r_lane_inds, _ = sliding_window_polyfit(img_bin)
#     else:
#         l_fit, r_fit, l_lane_inds, r_lane_inds = polyfit_using_prev_fit(img_bin, l_line.best_fit, r_line.best_fit)
        
    # invalidate both fits if the difference in their x-intercepts isn't around 350 px (+/- 100 px)
    if left_fit is not None and right_fit is not None:
        # calculate x-intercept (bottom of image, x=image_height) for fits
        h = img.shape[0]
        left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        x_int_diff = abs(right_fit_x_int-left_fit_x_int)
        if abs(350 - x_int_diff) > 100:
            left_fit = None
            right_fit = None
    
    left_line.add_fit(left_fit, left_lane_inds)
    right_line.add_fit(right_fit, right_lane_inds)
    
    # draw the current best fit if it exists
    if left_fit is not None and right_fit is not None and left_line.best_fit is not None and right_line.best_fit is not None:
        lan_img = draw_lane(new_img, binary_img)
        img_out = draw_data(lan_img, binary_img)
    else:
        img_out = new_img
    
    return img_out
    
```
---

### Run process_image on project video (input path: ../project_video_output.mp4):

![alt text][video_project_video_output]

### Run process_image on challenge video (input path: ../challenge_video.mp4):

![alt text][video_challenge_video_output]

### Run process_image on harder challenge video (input path: ../harder_challenge_video.mp4):

![alt text][video_harder_challenge_video_output]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problem I encountered is how to make good use of the two chapter's methods. Then I decided to create the function for method each by each and use testing images to check the results. By going through all the methods, I can fully understand how to detect the road lane. By referencing on the internet, I found someone used B channel of the LAB colorspace, which isolates the yellow lines very well. Therefore the main pipeline could be simplified to 4 steps (1) Undistort (2) Perspective Transform (3) use HLS L-channel Threshold to detect the while lines and (4) use LAB B-channel Threshold to detect the yellow lines. 

We could see some problem still exists in the video: (1) it would fail when the right white line is very short. (2) if the video light condition is more bright or dark, the current threshold settings could not detect the lines very well.
