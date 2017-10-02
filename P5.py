from Funcs import *
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from moviepy.editor import VideoFileClip
import time

ENABLE_PLOT_SHOW=False

#Step 1 Read Vehicle and non vehicle data
vehicle_ds=glob.glob('dataset/vehicles/**/*.png')
nonvehicle_ds=glob.glob('dataset/non-vehicles/**/*.png')
#
if ENABLE_PLOT_SHOW==True:
    print(len(vehicle_ds), len(nonvehicle_ds))

#Step 2 Check the best spatial color space 
if ENABLE_PLOT_SHOW==True:
    randomimage_V=cv2.imread(vehicle_ds[np.random.randint(0,len(vehicle_ds))])
    randomimage_NV=cv2.imread(nonvehicle_ds[np.random.randint(0,len(nonvehicle_ds))])
    testSpatialColor(randomimage_V)
    testSpatialColor(randomimage_NV,'Non-Vehicle')

#Step3 Test HOG 
if ENABLE_PLOT_SHOW==True:
    orient = 9
    pix_per_cell = 8
    cell_per_block = 8
    randomimage_V=cv2.imread(vehicle_ds[np.random.randint(0,len(vehicle_ds))])
    randomimage_NV=cv2.imread(nonvehicle_ds[np.random.randint(0,len(nonvehicle_ds))])
    _,out_V=get_hog_features(cv2.cvtColor(randomimage_V,cv2.COLOR_BGR2GRAY),orient,pix_per_cell, cell_per_block,vis=True)
    _,out_NV=get_hog_features(cv2.cvtColor(randomimage_NV,cv2.COLOR_BGR2GRAY),orient,pix_per_cell, cell_per_block,vis=True)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,7))
    ax1.set_title('Gray Vehicle')
    ax1.imshow(cv2.cvtColor(randomimage_V,cv2.COLOR_BGR2GRAY), cmap='gray')
    ax2.set_title('HOG Vehicle')
    ax2.imshow(out_V, cmap='gray')
    ax3.set_title('Gray Non Vehicle')
    ax3.imshow(cv2.cvtColor(randomimage_NV,cv2.COLOR_BGR2GRAY), cmap='gray')
    ax4.set_title('HOG Non Vehicle')
    ax4.imshow(out_NV, cmap='gray')
    plt.show()

# Step 4 Combination and prepare of DS 
svc=PrepareAndTrainSVM(vehicle_ds,nonvehicle_ds)

#Step 5 Use slide window to determine if the detected objected are vehicle/non vehicle
if ENABLE_PLOT_SHOW==True:
    testimages =glob.glob('./test_images/*.jpg')
    for path in testimages:
        im=mpimg.imread(path)
        ystart = 400
        ystop = 656
        scale = 1.5
        orient = 11
        pix_per_cell = 16
        cell_per_block = 2
        rects=find_cars(im,ystart,ystop,scale,'YUV','ALL',svc,orient,pix_per_cell,cell_per_block,None,None)
        out_im=np.copy(im)
        out_im=draw_boxes(out_im,rects)
        plt.imshow(out_im)
        plt.show()

#Step 6 Check different scale 

if ENABLE_PLOT_SHOW==True:
    testimages =glob.glob('./test_images/*.jpg')
    for path in testimages:
        im=mpimg.imread(path)
        rects=find_cars_differentScales(im,svc)
        out_im=applyheatmap(im,rectangles)
        plt.imshow(out_im)
        plt.show()

def process_image_video(image):
    rects=find_cars_differentScales(image,svc)
    out_image=applyheatmap(image,rects)
    return out_image

video_output = 'project_video_output.mp4'
video_input = VideoFileClip('project_video.mp4')
processed_video = video_input.fl_image(process_image_video)
processed_video.write_videofile(video_output, audio=False)

video_output2 = 'test_video_output.mp4'
video_input2 = VideoFileClip('test_video.mp4')
processed_video2 = video_input2.fl_image(process_image_video)
processed_video2.write_videofile(video_output2, audio=False)





