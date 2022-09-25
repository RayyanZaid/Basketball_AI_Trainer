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
# Working with Peter today

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
    h,w,_ = image.shape
    h = h // 2
    w = w // 2

    new_image = cv2.resize(image,(w,h))

    return new_image,h,w


# Pose processes the image parameter and returns the result image
# Create the pose process image function and test it out

def pose_process_image(image,pose):

    #return the original image AND the result image

    processed_image = pose.process(image)
    return image,processed_image
    
def plot_angles_from_frames(mp_pose,landmarks, image,h, w):
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

  angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_ELBOW.value, 
                            mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                            mp_pose.PoseLandmark.LEFT_HIP.value,landmarks, image,h, w+val)
  angles.append(angle)
  angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_ELBOW.value, 
                            mp_pose.PoseLandmark.RIGHT_SHOUDLER.value, 
                            mp_pose.PoseLandmark.RIGHT_HIP.value,landmarks, image,h, w-val)
  angles.append(angle)
  

  angle_wrist_shoulder_hip_left, image = plot_angle(mp_pose.PoseLandmark.LEFT_WRIST.value,
                                          mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                          mp_pose.PoseLandmark.LEFT_HIP.value,landmarks, image,h, w+val)

  angle_wrist_shoulder_hip_right, image = plot_angle(mp_pose.PoseLandmark.RIGHT_WRIST.value,
                                          mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                          mp_pose.PoseLandmark.RIGHT_HIP.value,landmarks, image,h, w-val)

  angles.append(angle_wrist_shoulder_hip_left)
  angles.append(angle_wrist_shoulder_hip_right)


  return angles


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

def draw_landmarks(results, mp_drawing,mp_pose,image):
   # do not display hand, feet
  for idx, landmark in enumerate(results.pose_landmarks.landmark):
    if idx in [1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32]:
        results.pose_landmarks.landmark[idx].visibility = 0
  
  mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
  return image

def get_frames_angles(image_name:str, video_path:str)->tuple:
  mp_pose, mp_drawing = set_up_pose_detection_model()
  cap = cv2.VideoCapture(video_path)
  out = get_video_writer(image_name,video_path)
  img_count = 1
  output_images = []
  frames = []

  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      success, image = cap.read()
    
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break
      image,h,w = resize_image(image)
       
      image, results =pose_process_image(image, pose)
      cv2.imshow('Image',image)
      
      try:
        landmarks = results.pose_landmarks.landmark
        angles = plot_angles_from_frames(mp_pose,landmarks, image,h, w)
        frames.append(angles)

      except:
        print('Failed')
      
      if cv2.waitKey(5) & 0xFF == 27:
        break


  
  
  return frames




# coach_frames, max_angle_right = get_frames_angles(image_name= 'coach', video_path="coach1.mp4")
# coach_frames = add_stage(coach_frames, max_angle_right)
student_frames,max_angle_right = get_frames_angles(image_name= 'student', video_path="coach1.mp4")



student_n_cluster = 4
print(student_n_cluster)
X = np.array(student_frames)
kmeans_student = KMeans(n_clusters=student_n_cluster, random_state=0).fit(X)
print(kmeans_student.labels_)