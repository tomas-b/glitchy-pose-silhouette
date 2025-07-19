
Search
Write
Sign up

Sign in



Introduction to Motion Detection: Part 3
How to use Background Subtraction to Detect Motion
Isaac Berrios
Isaac Berrios

Follow
5 min read
·
Nov 2, 2023
166


5



In the final part of this series, we will utilize Background Subtraction to detect motion, and we will see how it can outperform deep object detectors such as YOLO in terms of detecting object size. Code for this tutorial is on GitHub.

In case you missed it, here are the links to the rest of the tutorial.

Part 1 — Frame Differencing
Part 2 — Optical Flow
Part 3 — Background Subtraction

Photo by Ralph Mayhew on Unsplash
Introduction
In part 1 we used Frame Differencing and in part 2 we used Optical Flow to detection motion. Both of these methods are simple and provide decent (yet noisy) results. Let’s see how background subtraction can improve motion detection, the algorithm is outlined below:

Initialize Background Model
Update Background Model with new image frame
Obtain Foreground Mask from Background Model
Obtain Clean Motion Mask from Foreground Mask
Obtain initial motion detections from Motion Mask
Perform Non-Maximal Suppression on initial motion detections
Repeat steps 2–6
In this post we will cover the details of the algorithm and code it in Python from scratch.

Background Subtraction
In Background Subtraction, a learned background model is subtracted from the current frame to obtain a foreground mask.


Figure 1. Concept of Background Subtraction. Source.
The Background Modeling consists of two main steps:

Background Initialization
Background update
The background model becomes more accurate as more images are fed into. We will leverage two powerful Background models in opencv: MOG2 and KNN. MOG2 stands for Mixture of Gaussians 2 (it is an improved approach from MOG) and KNN stands for K Nearest Neighbors, the details of these models are out of scope for this post. We can perform step 1 and initialize the background model in Python with:

# get background subtractor
sub_type = 'KNN' # 'MOG2'
if sub_type == "MOG2":
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True)
    # backSub.setShadowThreshold(0.75)
else:
    backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=True)
Each Background model contains an option to detect shadows, the shadow threshold must be tuned. The overall effectiveness of shadow detection seems to depend on the data (sometimes it’s effective and sometimes it’s not). For simplicity, we will not try to detect shadows in this post.

Get Isaac Berrios’s stories in your inbox
Join Medium for free to get updates from this writer.

Enter your email
Subscribe
The code below shows how to obtain background and foreground estimates after 5 frames. Notice how we are able to update the background model and obtain the foreground mask with a single line of code (this takes care of steps 2 and 3 in the algorithm).

for img_path in image_paths[:5]:
    image = cv2.imread(img_path)
    # update the background model and obtain foreground mask
    fg_mask = backSub.apply(image)

# display
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].imshow(backSub.getBackgroundImage())
ax[0].set_title(f"Current {sub_type} Background after 5 frames")
ax[1].imshow(fg_mask, cmap='gray') 
ax[1].set_title(f"{sub_type} Foreground Mask after 5 frames");

Figure 2. Right: Background Model. Left: Foreground Mask. Source: Author.
Even after five frames, the background model is able to give us an accurate foreground mask. The snippet below shows how to process the foreground mask to get a clean motion mask. (If we want to threshold the shadows, we can set min_thresh to 127).

def get_motion_mask(fg_mask, min_thresh=0, kernel=np.array((9,9), dtype=np.uint8)):
    """ Obtains image mask
        Inputs: 
            fg_mask - foreground mask
            kernel - kernel for Morphological Operations
        Outputs: 
            mask - Thresholded mask for moving pixels
        """
    _, thresh = cv2.threshold(fg_mask,min_thresh,255,cv2.THRESH_BINARY)
    motion_mask = cv2.medianBlur(thresh, 3)
    
    # morphological operations
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return motion_mask

Figure 3. Motion Mask. Source: Author.
Now that we have the motion mask, we can obtain non-max suppressed detections using the approaches from part 1 (This covers steps 5 and 6 in the algorithm).


Figure 4. Non-Max Suppressed Bounding Boxes. Source: Author.
Create Detection Pipeline
Now we can streamline this approach into a single function.

def get_detections(backSub, frame, bbox_thresh=100, nms_thresh=0.1, kernel=np.array((9,9), dtype=np.uint8)):
    """ Main function to get detections via Frame Differencing
        Inputs:
            backSub - Background Subtraction Model
            frame - Current BGR Frame
            bbox_thresh - Minimum threshold area for declaring a bounding box
            nms_thresh - IOU threshold for computing Non-Maximal Supression
            kernel - kernel for morphological operations on motion mask
        Outputs:
            detections - list with bounding box locations of all detections
                bounding boxes are in the form of: (xmin, ymin, xmax, ymax)
        """
    # Update Background Model and get foreground mask
    fg_mask = backSub.apply(frame)

    # get clean motion mask
    motion_mask = get_motion_mask(fg_mask, kernel=kernel)

    # get initially proposed detections from contours
    detections = get_contour_detections(motion_mask, bbox_thresh)

    # separate bboxes and scores
    bboxes = detections[:, :4]
    scores = detections[:, -1]

    # perform Non-Maximal Supression on initial detections
    return non_max_suppression(bboxes, scores, nms_thresh)
The snippet below shows how can perform moving object detection on a video sequence.

kernel=np.array((9,9), dtype=np.uint8)

sub_type = 'MOG2' # 'KNN'
if sub_type == "MOG2":
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True)
    backSub.setShadowThreshold(0.5)
else:
    backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=True)

for idx in range(0, len(image_paths)):
    # read frames
    frame_bgr = cv2.imread(image_paths[idx])

    # get detections
    detections = get_detections(backSub, 
                                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), 
                                bbox_thresh=100, 
                                nms_thresh=1e-2, 
                                kernel=kernel)
                                
    # draw bounding boxes on frame
    draw_bboxes(frame_bgr, detections)

    # save image for GIF
    fig = plt.figure(figsize=(15, 7))
    plt.imshow(frame_bgr)
    plt.axis('off')
    fig.savefig(f"temp/frame_{idx}.png")
    plt.close();
Now we can display the obtained background models for MOG2 and KNN after only 90 frames.


Figure 5. Background Models obtained with MOG2 (right) and KNN (left). Source: Author.
These results aren’t bad, in this case MOG2 is clearly better, but both will continue to improve as more frames are added. Here are GIFs of the detections with each model.


Figure 6. Moving Object Detections with KNN Background Subtraction. Source: Author.

Figure 7. Moving Object Detections with MOG2 Background Subtraction. Source: Author.
If we look closely the detections take a few frames to appear, this is because it takes a few frames (~5) for the background models to learn the background. Background Subtraction provides a much better approach to Moving Object Detection than Frame Differencing or Optical Flow.

Conclusion
With Background Subtraction, we are able to detect much smaller objects in the video, even smaller than what a deep object detector such as YOLO would be able to detect. However, this comes with the consequence that there are many spurious small detections that are false. We can mitigate against this by using an object tracking algorithm.

Part 1 — Frame Differencing
Part 2 — Optical Flow
Part 3 — Background Subtraction
Thanks for reading all the way to the end! If you liked please consider clapping. If you ready to move onto something more advanced, see Introduction Unsupervised Motion Detection.

Ready to join the cutting edge?
Receive Private Daily Emails
Receive my Private Daily Emails, and learn to build self-driving cars, autonomous drones, robotics, and advanced…
t.dripemail2.com

Computer Vision
Object Detection
Opencv
Python
Background Subtraction
166


5


Isaac Berrios
Written by Isaac Berrios
626 followers
·
200 following
Electrical Engineer interested in Sensors and Autonomy. I write about Sensors, Signal Processing, Computer Vision, and Embedded Computing.


Follow
Responses (5)

Write a response

What are your thoughts?

Cancel
Respond
Isaac Berrios
Isaac Berrios

Author
Jan 31


I think this could work if you don't need a super high franw rate, but I haven't tried it on an embedded system. The frame differencing in part 1 might be better though
Reply

Sridevi Voleti
Sridevi Voleti

Dec 30, 2024


This is great content. I loved the part 1 & part 2, frame differencing has really set the ground for the subsequent parts. Thanks much for that.
However, for the Part 3, the GitHub repo seems to be broken.
I am unable to follow thru the code in GitHub.
10


1 reply

Reply

Daniel García
Daniel García

Mar 27, 2024


Great content!
10

Reply

See all responses
More from Isaac Berrios
Introduction to Motion Detection: Part 1
Isaac Berrios
Isaac Berrios

Introduction to Motion Detection: Part 1
The easiest way to detect motion with opencv
Oct 30, 2023
209
1
Introduction to Beamforming: Part 1
Isaac Berrios
Isaac Berrios

Introduction to Beamforming: Part 1
How to estimate the Direction of Arrival with an Array
Jul 8, 2024
38
DeepLabv3
Isaac Berrios
Isaac Berrios

DeepLabv3
Building Blocks for Robust Segmentation Models
May 30, 2023
94
3
Introduction to Beamforming: Part 3
Isaac Berrios
Isaac Berrios

Introduction to Beamforming: Part 3
How to derive and implement the Capon Beamformer
Aug 26, 2024
25
See all from Isaac Berrios
Recommended from Medium
Canny Edge Detection
The ByteDoodle Blog
In

The ByteDoodle Blog

by

Hareesha Dandamudi

Canny Edge Detection
Image Processing with OpenCV

Apr 1
Fundamentals of Image Processing in Python Using OpenCV
Sajid Khan
Sajid Khan

Fundamentals of Image Processing in Python Using OpenCV
How Computers See the World. Resizing and Grayscale, Edge Detection using Canny Algorithm, Image Thresholding example

Mar 7
16
YOLOv13 Just Changed Object Detection — Here’s What Makes It Different
Harish K
Harish K

YOLOv13 Just Changed Object Detection — Here’s What Makes It Different
To tell the truth, I was unsure when I first learned about YOLOv13. Over the years, YOLO has undergone so many iterations that it’s easy…

Jul 7
Building Vision Transformer: Deep Understanding, Building from Scratch and Hands-On PyTorch - Part…
AI Advances
In

AI Advances

by

Ranjeet Tiwari | Senior Architect - AI | IITJ

Building Vision Transformer: Deep Understanding, Building from Scratch and Hands-On PyTorch - Part…
The idea of slicing images into patches and feeding them to a Transformer sounded wild. Could I really build a ViT myself, without copying…

Jun 29
156
YOLOE: Revolutionizing Object Detection with Visual Prompts [Part-1]
akhil pillai
akhil pillai

YOLOE: Revolutionizing Object Detection with Visual Prompts [Part-1]
Introduction

Mar 31
7
Fundamentals of Image Processing and Computer Vision
ImageCraft
In

ImageCraft

by

Francisco Zavala

Fundamentals of Image Processing and Computer Vision
Vision is the most advanced of our senses. Thanks to it, we are able to orient ourselves in complex environments, recognize the difference…

Jun 10
85
See more recommendations
Help

Status

About

Careers

Press

Blog

Privacy

Rules

Terms

Text to speech
