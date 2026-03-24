import os
import json
import shutil

import cv2 as cv
import numpy as np
import pandas as pd
import mediapipe as mp

from tqdm import tqdm

# Class Imports
from cleaning.Normalizer import *
from plotting.Annotation import Annotater
from assingment.KeypointExtractor import KeypointExtractor


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def load_JSON_object(json_path: str):
    ''' Loads JSON object from a file
    '''
    with open(json_path, "r") as f:
        json_data = json.load(f)
        return json_data
    

def horizontally_flip_dataset(video_frame_folder: str, output_path: str ='data/Mirrored_video_frames/'):
    ''' Funtction for flipping dataset and updating resulting action
    Links:
        How to flip image : https://www.opencvhelp.org/tutorials/image-processing/how-to-flip-image/
        How to write to a file correctly : https://stackoverflow.com/questions/6159900/correct-way-to-write-line-to-file
    Args:
        dataset_path (String) : String path to folder whose contents we want to flip horizontally
        output_path (String) : File path to save folder of horizontally flipped images to
    '''
    # Iterate through dataset and horizontally flip all images to double dataset size
    create_folder(output_path)
    create_folder('data/mirrored_videos')
    for swing_type in os.listdir(video_frame_folder):
        create_folder(f'{output_path}{swing_type}/')
        create_folder(f'data/mirrored_videos/{swing_type}/')
        for uid_folder in tqdm(os.listdir(f'{video_frame_folder}{swing_type}'), desc=swing_type):
            dataset_path = f'{video_frame_folder}{swing_type}/{uid_folder}/'
            create_folder(f'{output_path}{swing_type}/{uid_folder}/')
            for file_name in os.listdir(dataset_path):
                # Flip and save image
                image = cv.imread(f'{dataset_path}{file_name}')
                flipped_image = cv.flip(image, 1)
                cv.imwrite(f'{output_path}{swing_type}/{uid_folder}/Mirrored_{file_name}', flipped_image)
            
            annotater = Annotater([])
            annotater.output_frames_to_video(frame_folder_path=f'{output_path}{swing_type}/{uid_folder}/',
                                             output_video_path = f'data/mirrored_videos/{swing_type}/',
                                             output_video_name = f'mirrored_{uid_folder}.mp4',
                                             fps=30, range=None)
    



def extract_pose_data(dataset_folder: str):
    ''' Iterates thorugh the video dataset folder and create a folder of pose JSON
    Args:
        dataset_folder (String) : String file path to fodler containing Guenoles dataset of tennis swings
    '''

    for swing_type in os.listdir(dataset_folder):
        # Extratc JSON from all video files
        for file in tqdm(os.listdir(f'{dataset_folder}{swing_type}/')):
            print(f'Extracting : {file}')
            create_folder(f'data/video_frames/{swing_type}/')
            create_folder(f'data/json_data/{swing_type}/')
           
            extractor = KeypointExtractor(video_frame_path=f'data/mirrored_video_frames/{swing_type}/', 
                                          output_json_path=f'data/mirrored_json_data/{swing_type}/', 
                                          method='norm')
            extractor.video_to_json(f'{dataset_folder}{swing_type}/{file}')
        print("FINISHED")



def plot_all_annotations(json_folder_path, frame_folder_path, output_folder_base=None):
    ''' Folder to plot all skeletal data over the annotations
    Args:
        json_folder_path (String) : String file path to folder of all swing JSON
        frame_folder_path (String) : String file path to folder of all video frames
        output_folder_base (String) : String file path to base folder to store annoatted folders of frames
    '''
    create_folder(output_folder_base)
    for swing_type in os.listdir(json_folder_path):
        # Create folder to store all annotated folders of frames for that wsing type
        create_folder(f'{output_folder_base}{swing_type}/')
        
        for file in tqdm(os.listdir(f'{json_folder_path}{swing_type}/')):
            # Load JSON data
            json_data = load_JSON_object(f'{json_folder_path}{swing_type}/{file}')
            file_frame_folder_path = f'{frame_folder_path}{swing_type}/{file.split(".")[0]}/'
            output_folder_path = f'{output_folder_base}{swing_type}/{file.split(".")[0]}/'

            # Get image dimensions
            file_list = os.listdir(file_frame_folder_path)
            img = cv.imread(f'{file_frame_folder_path}{file_list[0]}')
            height, width, _ = img.shape


            create_folder(output_folder_path)
            # REMOVE AT SOME POINT
            gender = "u"
            session_id = "123456"
            count = 1
            age_upper = 8
            age_lower = 8
            fps = 30
            screen_width = width
            screen_height = height
            dm = Normalizer(json_data, session_id, count, gender, age_upper, age_lower, fps, screen_width, screen_height)

            # print('json_folder_path:', json_folder_path)
            # print('frame_folder_path:', frame_folder_path)
            # print('output_folder_base:', output_folder_base)
            # a-b
            annotater = Annotater(dm.cleaned_df)
            annotater.annotate_frames(file_frame_folder_path, 
                              dm.cleaned_df,
                              output_folder_path,
                              ratios=[width, height])



def build_all_folders():
    ''' Helper function for building all neccesary folders for extracting JSON data
    '''
    create_folder('data/')
    create_folder('data/video_frames/')
    create_folder('data/json_data/')


def create_frame_movement_dfs(dataset_path):
    for swing_type in os.listdir(dataset_path):
        df = pd.DataFrame()
        df['video_name'] = [file.split(".")[0] for file in os.listdir(f'{dataset_path}{swing_type}')]
        create_folder('data/csv/')
        df.to_csv(f'data/csv/{swing_type}.csv')



def create_mirrored_videos(frame_path, output_path):
    ''' Stitch folders of mirrored frames back together 
    Links:
        stitch frames : https://stackoverflow.com/questions/43048725/python-creating-video-from-images-using-opencv
    Args:
        frame_path (String) : 
        output_path (String) :
    '''
    for swing_type in os.listdir(frame_path):
        create_folder(f'{output_path}{swing_type}/')
        for frame_folder in tqdm(os.listdir(f'{frame_path}{swing_type}/'), desc=swing_type):
            # Sort frames numerically
            frame_nums = {int(file_name.split(".")[0].replace("Mirrored_", "")) : file_name for file_name in os.listdir(f'{frame_path}{swing_type}/{frame_folder}/')}
            sorted_frames = sorted(frame_nums.keys())

            # Get image dimensions
            img = cv.imread(f'{frame_path}{swing_type}/{frame_folder}/{frame_nums[0]}')
            height, width, _ = img.shape

            # write frames back to video
            fourcc = cv.VideoWriter_fourcc(*'mp4v') 
            video = cv.VideoWriter(f'{output_path}{swing_type}/Mirrored_{frame_folder}.mp4', fourcc, 30, (width, height))
            for frame_idx in sorted_frames:
                img = cv.imread(f'{frame_path}{swing_type}/{frame_folder}/{frame_nums[frame_idx]}')
                video.write(img)

            cv.destroyAllWindows()
            video.release()


def get_total_action_counts(json_folders):
    ''' Get total action count of all actions in dataset
        Replace passing args with list of file paths for when I added synthesized subsampled data
    Args:
        json_folder_path () :
        mirrored_json_folder_path () :
    '''
    action_set = {'Backhand' : 0, 'Forehand' : 0, 'Serves' : 0, 'NoStroke' : 0}
    print(action_set)
    for json_folder in json_folders:
        for action in action_set:
            print(f'FOLDER : {json_folder} : ACTION: {action}, LEN : {len(os.listdir(f"{json_folder}{action}"))}...')
    a-b

def main():
    ''' 
        Plan:
            Extract JSON from all files
            Clean JSON
            See how often Skeleton swpas between people
    '''
    # build_all_folders()

    video_folder_path = '../VideoDataset/'
    # extract_pose_data(video_folder_path)

    # Plot skeleton annotations
    json_folder_path = 'data/json_data/raw/'
    frame_folder_path = 'data/video_frames/raw/'
    annotated_frame_folder = 'data/video_frames/annotated_raw_frames/'
    
    mirrored_json_folder_path = 'data/json_data/mirrored_json_data/'
    mirrored_video_folder_path = 'data/videos/mirrored_videos/'
    mirrored_frame_folder_path = 'data/video_frames/mirrored_video_frames/'
    mirrored_annotated_frame_folder = 'data/video_frames/mirrored_annotated_frames/'
    
    # NOTE - Preprocessing steps --> Flipping dataset, dataset synthesisation using mismatched frames
    # Horizontally flip dataset
    # horizontally_flip_dataset(frame_folder_path)

    # Stictch frames back together to make mirrored videos
    # create_mirrored_videos(mirrored_frame_folder_path, mirrored_video_folder_path)

    # NOTE - Extract pose data on flipped videos
    # extract_pose_data('data/videos/')
    # # extract_pose_data('data/mirrored_videos/')

    # NOTE - Plot annotations back over images as sanity test
    # plot_all_annotations(json_folder_path, frame_folder_path, output_folder_base=annotated_frame_folder)
    # plot_all_annotations(mirrored_json_folder_path, mirrored_frame_folder_path, output_folder_base=mirrored_annotated_frame_folder)

    # Get total action instance count in dataset
    json_folders = [json_folder_path, mirrored_json_folder_path]
    get_total_action_counts(json_folders)
    

if __name__ == "__main__":
    main()