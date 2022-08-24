from email.mime import base
import cv2
import mediapipe as mp
import numpy as np
import sys 
import os
# matplotlib (later)
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans


# Create function that sets up our mediapipe pose model


def set_up_pose_detection_model():
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_pose = mp.solutions.pose
  return mp_pose, mp_drawing



def make_directory(name:str):
    if not os.path.isdir(name):
        os.mkdir(name)
    

# cv2.VideoWriter
def get_video_writer(image_name,video_path):
    basename = os.path.basename(video_path)
    filename, extension = os.path.splitext(basename)
    size = (480,640)
    make_directory(image_name)
    out = cv2.VideoWriter(f"{image_name}/{filename}_out.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 5, size)
    return out

def resize_image(image):
    # Get height and width of image
    # Resize it to half its height and width 
    # Return 3 things (new image, new height, new width)
    h,w = image.shape
    h = h // 2
    w = w // 2

    new_image = cv2.resize(image,(w,h))

    return new_image,h,w


# Pose processes the image parameter and returns the result image
# Create the pose process image function and test it out

def pose_process_image(image,pose):

    #return the original image AND the result image

    results = pose.process(image)
    return image,results
    
def plot_angles_from_frames(mp_pose,landmarks, image,h, w, max_angle_right = 0, ):
  angles = []
  val =  50
  angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                            mp_pose.PoseLandmark.LEFT_ELBOW.value,
                            mp_pose.PoseLandmark.LEFT_WRIST.value,landmarks, image,h, w + val)
  angles.append(angle)
  angle, image =plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                            mp_pose.PoseLandmark.RIGHT_ELBOW.value, 
                            mp_pose.PoseLandmark.RIGHT_WRIST.value,landmarks, image,h, w - val)
  angles.append(angle)
  angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_HIP.value,
                            mp_pose.PoseLandmark.LEFT_KNEE.value,
              mp_pose.PoseLandmark.LEFT_ANKLE.value,landmarks, image,h, w+val)
  angles.append(angle)
  angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_HIP.value,
                            mp_pose.PoseLandmark.RIGHT_KNEE.value,
          mp_pose.PoseLandmark.RIGHT_ANKLE.value,landmarks, image,h, w-val)
  angles.append(angle)

  angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                            mp_pose.PoseLandmark.LEFT_HIP.value, 
                            mp_pose.PoseLandmark.LEFT_KNEE.value,landmarks, image,h, w+val)
  angles.append(angle)
  angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
                            mp_pose.PoseLandmark.RIGHT_HIP.value, 
                            mp_pose.PoseLandmark.RIGHT_KNEE.value,landmarks, image,h, w-val)
  angles.append(angle)
  

  angle_wrist_shoulder_hip_left, image = plot_angle(mp_pose.PoseLandmark.LEFT_WRIST.value,
                                          mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                          mp_pose.PoseLandmark.LEFT_HIP.value,landmarks, image,h, w+val)

  angle_wrist_shoulder_hip_right, image = plot_angle(mp_pose.PoseLandmark.RIGHT_WRIST.value,
                                          mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                          mp_pose.PoseLandmark.RIGHT_HIP.value,landmarks, image,h, w-val)

  angles.append(angle_wrist_shoulder_hip_left)
  angles.append(angle_wrist_shoulder_hip_right)
  max_angle_right = max(max_angle_right, angle_wrist_shoulder_hip_right) 
  # print(max_angle_right)

  return angles, max_angle_right


def plot_angle(p1,p2, p3,landmarks, image,h, w):
    # Get coordinates
    a = [landmarks[p1].x,
                landmarks[p1].y]
    b= [landmarks[p2].x, landmarks[p2].y]
    c = [landmarks[p3].x, landmarks[p3].y]

    # Calculate angle
    angle = calculate_angle(a, b, c)
    # print(angle)
    draw_angle( tuple(np.multiply(b, [w, h]).astype(int)), image, round(angle))
    return angle, image


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle,1)

def draw_angle(org :tuple, image, angle):
  # font
  font = cv2.FONT_HERSHEY_SIMPLEX
  # fontScale
  fontScale = 0.4
  # Blue color in BGR
  color = (255, 255, 255)
    
  # Line thickness of 2 px
  thickness = 1
    
  # Using cv2.putText() method
  image = cv2.putText(image, str(angle), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
  return image
