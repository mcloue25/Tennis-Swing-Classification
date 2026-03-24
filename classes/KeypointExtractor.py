import os
import cv2
import json
import shutil
import base64
import ffmpeg
import random

import numpy as np

from mediapipe.python.solutions import pose as mp_pose 
from mediapipe.python.solutions import drawing_utils as mp_drawing

class KeypointExtractor(object):
    '''Class to handle keypoint data extraction from video.
    '''
    def __init__(self, video_frame_path : str, output_json_path : str, method : str):
        ''' Constructor for KeypointExtractor Class.
        Args: 
            video_frame_path (String): Path to the folder to store video frames.
            output_json_path (String): Path to the folder to store pose JSON.
            method (String): Method to use when calcualting pose coordiantes - 'norm' or 'world'.
        Returns:
            KeypointExtractor (KeypointExtractor) = Instance of the KeypointExtractor class.
        '''
        self.video_frame_path = video_frame_path
        self.output_json_path = output_json_path

        methods = ("world", "norm", "screen")
        if method not in methods:
            raise ValueError(f"method: status must be one of {methods}.")
        else:
            self.method = method

        # Create folders for video frame and JSON output
        if not os.path.exists(video_frame_path):
            os.makedirs(video_frame_path)

        if not os.path.exists(output_json_path):
            os.makedirs(output_json_path)

        # Load pose estimation model.
        # mp_pose = mp.solutions.pose
        # self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True, smooth_segmentation=True)



    def convert_video_to_images(self, video_path, video_name):
        ''' Takes a video as input and splits that video into a number of frames.
        Args:
            video_path (String): The file path to the video we wish to split.
            output_path(String) : The path at which to save the video frames.  
        '''
        vidcap = cv2.VideoCapture(video_path)
        self.fps = int(vidcap.get(cv2.CAP_PROP_FPS))  

        self.current_frame_path = f"{self.video_frame_path}/{video_name}"

        if not os.path.exists(self.current_frame_path):
            os.mkdir(self.current_frame_path)

        count = 0
        success = True
        while success:
            success, image = vidcap.read()
            if success:
                cv2.imwrite(self.current_frame_path+f"/{count}.jpg", image)
                count += 1
    

    def process_frames(self, video_name, json_path):
        ''' Takes a file path to a folder of frames and generates a JSON object consisting of keypoint data for each frame.
        Args:
            video_name (String): Name of the video being processed.
        '''
        split_file_name = video_name.split("_")
        keypoints = ['NOSE','LEFT_EYE_INNER','LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR','RIGHT_EAR','MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_ELBOW','RIGHT_ELBOW','LEFT_WRIST','RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
        
        frame_data = []
            
        # Order frames & extract image_width & image_height
        file_list = os.listdir(self.current_frame_path)
        file_list.sort(key=lambda line: int(line.split(".")[0]))
        image_height, image_width, _ = cv2.imread(f'{self.current_frame_path}/{file_list[0]}').shape

        time_delta = "{:.4f}".format(float(float(self.fps) / len(file_list)))

        # Iterate through file list to get keypoint data fro each frame
        for frame_index, file in enumerate(file_list):
            part_keypoints = []   
            image = cv2.imread(f'{self.current_frame_path}/{file}')
            
            # Convert the BGR image to RGB before processing.
            results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # No data picked up by the model, set the score to 0
            if not results.pose_landmarks:
                score = 0
            else:
                score = 1
            
            try:
                # Iterate through each keypoint and get the X & Y values
                if self.method == "screen": # IMAGE SPACE
                    for index, keypoint in enumerate(keypoints):
                        # Getting the X & Y for each point
                        x = results.pose_landmarks.landmark[index].x
                        y = results.pose_landmarks.landmark[index].y
                        z = results.pose_landmarks.landmark[index].z
                        visibility = results.pose_landmarks.landmark[index].visibility

                        # Y is inverted due to different coordinate scale used by MediaPipe Pose
                        converted_point = {"part": keypoint, "x": x, "y": 1 - y, "z" : z, "w": visibility}
                        part_keypoints.append(converted_point)

                if self.method == "norm": # IMAGE SPACE
                    for index, keypoint in enumerate(keypoints):
                        # Getting the X & Y for each point
                        x = results.pose_landmarks.landmark[index].x
                        y = results.pose_landmarks.landmark[index].y
                        z = results.pose_landmarks.landmark[index].z
                        visibility = results.pose_landmarks.landmark[index].visibility

                        # Y is inverted due to different coordinate scale used by MediaPipe Pose
                        converted_point = {"part": keypoint, "x": x, "y": 1 - y, "z" : z, "w": visibility}
                        part_keypoints.append(converted_point)

                if self.method == "world":
                    for index, keypoint in enumerate(keypoints):
                        # Getting the X & Y for each point
                        x = results.pose_world_landmarks.landmark[index].x
                        y = results.pose_world_landmarks.landmark[index].y
                        z = results.pose_world_landmarks.landmark[index].z
                        visibility = results.pose_world_landmarks.landmark[index].visibility

                        # Y is inverted due to different coordinate scale used by MediaPipe Pose
                        converted_point = {"part": keypoint, "x": x, "y": 1 - y, "z" : z, "w": visibility}
                        part_keypoints.append(converted_point)

                # Used to store all data for a single frame
                builder = {"score": score, "frame_index" : frame_index, "time": time_delta, "key_points" : part_keypoints}
                frame_data.append(builder)

            # Except block to catch NoneType errors for empty frames
            except AttributeError: 
                for index, keypoint in enumerate(keypoints):
                    converted_point = {"part": keypoint, "x": np.nan, "y": np.nan, "z" : np.nan, "w": 0}
                    part_keypoints.append(converted_point)

                # Used to store all data for a single frame
                builder = {"score" : score, "frame_index" : frame_index, "time": time_delta, "key_points" : part_keypoints}
                frame_data.append(builder)

        # Building the total JSON file
        new_json = {"screen_width" : image_width, 'screen_height' : image_height, "list_body_part_points": frame_data}   

        # Write the JSON to the desired file 
        with open(json_path, mode='w+') as json_file:
            json.dump(new_json, json_file)



    def process_frames_norm_world(self, video_name, json_path):
        ''' Takes a file path to a folder of frames and generates a JSON object consisting of keypoint data for each frame.
        Args:
            video_name (String): Name of the video being processed.      
        '''
        split_file_name = video_name.split("_")
        keypoints = ['NOSE','LEFT_EYE_INNER','LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR','RIGHT_EAR','MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_ELBOW','RIGHT_ELBOW','LEFT_WRIST','RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
        
        frame_data = []
            
        # Order frames & extract image_width & image_height
        file_list = os.listdir(self.current_frame_path)
        file_list.sort(key=lambda line: int(line.split(".")[0]))
        image_height, image_width, _ = cv2.imread(f'{self.current_frame_path}/{file_list[0]}').shape

        time_delta = "{:.4f}".format(float(float(self.fps) / len(file_list)))

        # Iterate through file list to get keypoint data fro each frame
        for frame_index, file in enumerate(file_list):
            part_keypoints_norm = []
            part_keypoints_world = []   
            image = cv2.imread(f'{self.current_frame_path}/{file}')
            
            # Convert the BGR image to RGB before processing.
            results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # No data picked up by the model, set the score to 0
            if not results.pose_landmarks:
                score = 0
            else:
                score = 1
            
            try:
                for index, keypoint in enumerate(keypoints):
                    # Getting the X & Y for each point
                    x_norm = results.pose_landmarks.landmark[index].x
                    y_norm = results.pose_landmarks.landmark[index].y
                    z_norm = results.pose_landmarks.landmark[index].z
                    visibility_norm = results.pose_landmarks.landmark[index].visibility

                    # Y is inverted due to different coordinate scale used by MediaPipe Pose
                    converted_point_norm = {"part": keypoint, "x": x_norm, "y": 1 - y_norm, "z" : z_norm, "w": visibility_norm}
                    part_keypoints_norm.append(converted_point_norm)

                    # Getting the X & Y for each point
                    x_world = results.pose_world_landmarks.landmark[index].x
                    y_world = results.pose_world_landmarks.landmark[index].y
                    z_world = results.pose_world_landmarks.landmark[index].z
                    visibility_world = results.pose_world_landmarks.landmark[index].visibility

                    # Y is inverted due to different coordinate scale used by MediaPipe Pose
                    converted_point_world = {"part": keypoint, "x": x_world, "y": 1 - y_world, "z": z_world, "w": visibility_world}
                    part_keypoints_world.append(converted_point_world)

                # Used to store all data for a single frame
                builder = {"score": score, "frame_index" : frame_index, "time": time_delta, "key_points_norm": part_keypoints_norm, "key_points_world": part_keypoints_world}
                frame_data.append(builder)

            # Except block to catch NoneType errors for empty frames
            except AttributeError: 
                for index, keypoint in enumerate(keypoints):
                    converted_point_nan = {"part": keypoint, "x": np.nan, "y": np.nan, "z" : np.nan, "w": 0}
                    part_keypoints_norm.append(converted_point_nan)
                    part_keypoints_world.append(converted_point_nan)

                # Used to store all data for a single frame
                builder = {"score" : score, "frame_index" : frame_index, "time": time_delta, "key_points" : converted_point_nan}
                frame_data.append(builder)

        # Building the total JSON file
        new_json = {"screen_width" : image_width, 'screen_height' : image_height, "list_body_part_points": frame_data}   

        # Write the JSON to the desired file 
        with open(json_path, mode='w+') as json_file:
            json.dump(new_json, json_file)



    def video_to_json(self, video_path, video_name=''):
        ''' Function that handles splitting a video into frames, processing those frames and storing JSON data. 
        Args:
            video_path (String) : Path to the video to be processed.
        
        ''' 
        if video_name == "":
            try:
                split_path = video_path.split("/")
                video_name = split_path[-1].split(".")[0]
            except:
                video_name = video_path[-1].split(".")[0]
        
        # Convert the video to a series of frames
        self.convert_video_to_images(video_path, video_name)
        json_path = f"{self.output_json_path}/{video_name}.json"

        # Process frames and store JSON output
        # self.process_frames_norm_world(video_name, json_path)

        self.process_frames(video_name, json_path)

        return video_name, json_path
    

    def video_to_json_with_segmentation(self, video_path):
        ''' Function that handles splitting a video into frames, processing those frames and storing JSON data. 
        Args:
            video_path (String) : Path to the video to be processed.
        '''
        try:
            split_path = video_path.split("/")
            video_name = split_path[-1].split(".")[0]

        except:
            video_name = video_path[-1].split(".")[0]
        
        # Convert the video to a series of frames
        self.convert_video_to_images(video_path, video_name)
        json_path = f"{self.output_json_path}/{video_name}.json"

        # Process frames and store JSON output
        self.process_frames_with_segmentation(video_name, json_path)
        # self.process_frames(video_name, json_path)

        return video_name, json_path



    def annotate_image(self, image_path, annotated_name, skeleton_color=(0, 255, 0)):
        ''' Used to annotate an image with keypoint skeleton
        '''
        # Load pose estimation model.
        # mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2, static_image_mode=True)
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Convert the BGR image to RGB before processing.
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        annotated_image = image.copy()

        # mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=0, color=skeleton_color),
            connection_drawing_spec=mp_drawing.DrawingSpec(thickness=3, color=skeleton_color))

        cv2.imwrite("data/video_frames/annotated/images/{}.png".format(annotated_name), annotated_image) 