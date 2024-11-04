import cv2
import numpy as np
from moviepy import *
from IPython.display import HTML
from IPython.display import Image
import os
import glob
from moviepy.editor import VideoFileClip


def convert_to_hls(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2HLS)

def color_selection(image):
    hls=convert_to_hls(image)
    # for white lane
    lower_thresh= np.uint8([0,200,0])
    high_thresh=np.uint8([255,255,255])
    white_mask=cv2.inRange(hls,lower_thresh,high_thresh)

     #for yellow lane
    lower_thresh= np.uint8([15, 38, 115])
    high_thresh=np.uint8([40,255,255])
    yellow_mask=cv2.inRange(hls,lower_thresh,high_thresh)

    mask=cv2.bitwise_or(white_mask,yellow_mask)
    masked_og_img=cv2.bitwise_and(image,image,mask=mask)
    return masked_og_img

def canny(image):
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gaussian=cv2.GaussianBlur(gray,(5,5),0)
    canny_edges=cv2.Canny(gaussian,50,150)
    return canny_edges

def roi(image):
    mask=np.zeros_like(image)
    if len(image.shape)>2:
        channel_count=image.shape[2]
        color_to_be_ignored=(255,)*channel_count
    else:
        color_to_be_ignored=255
    

    height,width=image.shape[:2]
    bot_left=[width*0.1,height*0.95]
    bot_right=[width*0.9,height*0.95]
    top_left=[width*0.4,height*0.60]
    top_right=[width*0.6,height*0.6]

    vertices=np.array([[bot_left,bot_right,top_right,top_left]],dtype=np.int32)
    cv2.fillPoly(mask, vertices,color_to_be_ignored)
    masked_edges = cv2.bitwise_and(image, mask)
    return masked_edges

def houghTransform(image):
    
    rho = 1              
    theta = np.pi/180    
    threshold = 20      
    minLineLength = 20   
    maxLineGap = 300    
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)

def drawLine(image, lines, color = [0, 0, 255], thickness = 2):
    image = np.copy(image)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def avg_slope_itercept(lines):
    left_line=[]
    left_weight=[]
    right_line=[]
    right_weight=[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1==x2:
                continue
            slope=(y2-y1)/(x2-x1)
            intercept=y1-(slope*x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope<0:
                left_line.append((slope,intercept))
                left_weight.append(length)
            else:
                right_line.append((slope,intercept))
                right_weight.append(length)
    left_lane=np.dot(left_weight,left_line)/np.sum(left_weight)if len(left_weight)>0 else None
    right_lane=np.dot(right_weight,right_line)/np.sum(right_weight)if len(right_weight)>0 else None
    return left_lane,right_lane

def pixel_point(y1,y2,line):
    if line is None:
        return None
    slope,intercept=line
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    y1=int(y1)
    y2=int(y2)
    return ((x1,y1),(x2,y2))

def lane_line(image,lines):
    left_lane,right_lane=avg_slope_itercept(lines)
    y1=image.shape[0]
    y2=int(y1*0.6)
    left_line=pixel_point(y1,y2,left_lane)
    right_line=pixel_point(y1,y2,right_lane)
    return left_line,right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
  
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)



test_image=cv2.imread("./images/straight_lines2.jpg",cv2.IMREAD_COLOR)
cv2.imshow("test images", test_image)

colored_lane=color_selection(test_image)
cv2.imshow("colored lane",colored_lane)

canny_edges=canny(colored_lane)
cv2.imshow("Canny edges",canny_edges)

masked_region=roi(canny_edges)
cv2.imshow("masked region",masked_region)

hough=houghTransform(masked_region)
lines=drawLine(test_image,hough)
cv2.imshow("img with line",lines)

final=draw_lane_lines(test_image,lane_line(test_image,hough))
cv2.imshow("final",final)
cv2.waitKey(0)
cv2.destroyAllWindows()


def process_frame(image):
    col_selection=color_selection(image)
    edges=canny(col_selection)
    mask_region=roi(edges)
    linep=houghTransform(mask_region)
    result=draw_lane_lines(image,lane_line(image,linep))
    return result

def process_video(test_video, output_video):
    input_video = VideoFileClip(os.path.join('test_videos', test_video), audio=False)
    processed = input_video.fl_image(process_frame)
    processed.write_videofile(os.path.join('output_videos', output_video), audio=False)

# process_video(r'D:\Desktop\lane\video\test.mp4', 'test_output.mp4')
# HTML("""
# <video width="960" height="540" controls>
#   <source src="{0}">
# </video>
# """.format("output_videos\test_output.mp4"))


