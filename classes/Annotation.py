import cv2
import ffmpeg
import numpy as np
import os


# from moviepy.editor import VideoFileClip
from moviepy import VideoFileClip

# Colours for skeleton annotation
right_leg_colours = {
    (24, 26) : (255, 0, 0),
    (26, 28) : (0, 128, 255),
    (28, 32) : (255, 0, 255),
    (32, 30) : (0, 204, 0),
    (30, 28) : (153, 153, 0),
}

left_leg_colours = {
    (23, 25) : (255, 0, 0),
    (25, 27) : (0, 128, 255),
    (27, 31) : (255, 0, 255),
    (31, 29) : (0, 204, 0),
    (29, 27) : (153, 153, 0),
}

right_arm_colours = {
    (12, 14) : (255, 0, 0),
    (14, 16) : (0, 128, 255), 
    (16, 18) : (255, 0, 255),
    (16, 20) : (0, 204, 0),
    (16, 22) : (153, 153, 0)
}

left_arm_colours = {
    (11, 13) : (255, 0, 0),
    (13, 15) : (0, 128, 255), 
    (15, 17) : (255, 0, 255),
    (15, 19) : (0, 204, 0),
    (15, 21) : (153, 153, 0)
}

text_settings = {
    "font" : cv2.FONT_HERSHEY_SIMPLEX,
    "fontScale" : 1,
    "fontColor" : (0,255,0),
    "thickness" : 3,
    "lineType" : 2,
}

class Annotater:
    '''Class used to handle JSON annotation.
    '''
    def __init__(self, df, vars={}):
        ''' Constructor for Annotater class.
        Args:
            df (DataFrame) : Pandas DataFrame containing keypoint information.
            vars (Dict) : Dictionary containing key, value pairs for additional variables to be plotted.
        '''
        self.df = df
        self.vars = vars
        self.KEYPOINTS = [
                "NOSE",
                "LEFT_INNER_EYE",
                "LEFT_EYE",
                "LEFT_OUTER_EYE",
                "RIGHT_INNER_EYE",
                "RIGHT_EYE",
                "RIGHT_OUTER_EYE",
                "LEFT_EAR",
                "RIGHT_EAR",
                "LEFT_MOUTH",
                "RIGHT_MOUTH",
                "LEFT_SHOULDER",
                "RIGHT_SHOULDER",
                "LEFT_ELBOW",
                "RIGHT_ELBOW",
                "LEFT_WRIST",
                "RIGHT_WRIST",
                "LEFT_PINKY",
                "RIGHT_PINKY",
                "LEFT_INDEX",
                "RIGHT_INDEX",
                "LEFT_THUMB",
                "RIGHT_THUMB",
                "LEFT_HIP",
                "RIGHT_HIP",
                "LEFT_KNEE",
                "RIGHT_KNEE",
                "LEFT_ANKLE",
                "RIGHT_ANKLE",
                "LEFT_HEEL",
                "RIGHT_HEEL",
                "LEFT_FOOT",
                "RIGHT_FOOT"
        ]
        self.connections = [
            # the same as Upper Body 
            0, 1, 1, 2, 2, 3, 3, 7, 0, 4, 4, 5, 5, 6, 6, 8, 9, 10, 11, 12, 11, 13, 13, 15, 15, 17, 15, 19, 15, 21, 17, 19, 12, 14, 14, 16, 16, 18, 16, 20, 16, 22, 18, 20, 11, 23, 12, 24, 23, 24,
            # left leg
            24, 26, 26, 28, 28, 32, 32, 30, 30, 28,
            # right leg
            23, 25, 25, 27, 27, 31, 31, 29, 29, 27]
        # self.read_keypoints_from_df()

        
    def read_keypoints_from_df(self):
        '''Reads keypoints form the DataFrame for use in annotation.
        '''
        pose_kp_list = []
        for index, row in self.df.iterrows():
            pose_kp = np.zeros((33,2), np.float32)
            for index, kp in enumerate(self.KEYPOINTS):
                kp_x = f"{kp}_X"
                kp_y = f"{kp}_Y"
                # NOTE: Why is Y inverted? Test
                pose_kp[index, 0] = row[kp_x]
                pose_kp[index, 1] = 1 - row[kp_y]
            
            pose_kp_list.append(pose_kp)
        self.kps = pose_kp_list



    def draw_links_coloured(self, img, kps, connections, ratio=None):
        ''' Draw limbs to annotate pose skeleton.
        Args:
            img (Image) : Frame upon which limbs are annotated.
            kps (List) : List of keypoint coordinates.
            connections(List) : Points in which lines are drawn to represent limbs.
            ratio (Tuple) : Image dimensions to map normalised coordinates to image space.
        '''    
        left_leg = [(23, 25), (25, 27), (27, 31), (31, 29), (29, 27)]
        right_leg = [(24, 26), (26, 28), (28, 32), (32, 30), (30, 28)]
        right_arm = [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22)]
        left_arm = [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21)]

        color = (147,20,255)
        for i in range(0, len(connections), 2): 
            color = (147,20,255)
            if (connections[i], connections[i+1]) in right_leg:
                color = right_leg_colours[(connections[i], connections[i+1])]

            if (connections[i], connections[i+1]) in left_leg:
                color = left_leg_colours[(connections[i], connections[i+1])]
            
            if (connections[i], connections[i+1]) in right_arm:
                color = right_arm_colours[(connections[i], connections[i+1])]
            
            if (connections[i], connections[i+1]) in left_arm:
                color = left_arm_colours[(connections[i], connections[i+1])]

            cv2.line(img, (int(round(kps[connections[i],0]*ratio[0])), int(round(kps[connections[i],1]*ratio[1]))), (int(round(kps[connections[i+1],0]*ratio[0])), int(round(kps[connections[i+1],1]*ratio[1]))), color=color, lineType=cv2.LINE_AA, thickness=3)
      

    def convert_avi_to_mp4(self, input_avi_path, output_mp4_path):
        '''Convert AVI video to MP4 format.
        Args:
            kps (List) : List of keypoint coordinates.
            background_image_path (Str) : Path tot he image on which the skeleton is annotated.
        '''   
        try:
            # Load the AVI file
            clip = VideoFileClip(input_avi_path)
            # Write the video to an MP4 file
            clip.write_videofile(output_mp4_path, codec='libx264')
            print(f"Conversion successful: {output_mp4_path}")
            os.remove(input_avi_path)
        except Exception as e:
            print(f"An error occurred: {e}")

        return True



    def convert_video_to_images(self, video_path, output_folder_path):
        ''' Takes a video as input and splits that video into a number of frames.
        Args:
            video_path (String): The file path to the video we wish to split.
            output_path(String) : The path at which to save the video frames.
            
        '''
        vidcap = cv2.VideoCapture(video_path)
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)

        count = 0
        while (vidcap.isOpened()):
            success,image = vidcap.read()
            if not success:
                break

            meta_dict = ffmpeg.probe(video_path)
            rotateCode = None
            try:
                if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
                    rotateCode = cv2.ROTATE_90_CLOCKWISE
                elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
                    rotateCode = cv2.ROTATE_180
                elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
                    rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
            except Exception:
                pass

            if not rotateCode == None:
                image = cv2.rotate(image, rotateCode)
                cv2.imwrite(output_folder_path + "/%d.jpg" % count, image)
                count += 1   
            else:
                cv2.imwrite(output_folder_path + "/%d.jpg" % count, image)
                count += 1

        return True


    def annotate_frames(self, frame_folder_path, df, output_folder_path, ratios=[1080, 1920], resize=True):
        '''Overlay skeleton on original video.

        Args:
            frame_folder_path (Str) : Path to folder containing video frames.
            df (DataFrame) : DataFrame containing keypoint coordinates.
            output_folder_path (Str) : Path to folder for saving annotated frames.
            ratios (Tuple) : Width and Height of original video.
            resize (Bool) : Bool to decide whether or not to resize annotated frames.
            anonymise (Bool) : Bool to decide whether or not to anonymise the subjects face.

        ''' 
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)
        
        # Saves results to self.kps
        self.read_keypoints_from_df()
        frame_names = os.listdir(frame_folder_path)

        # Update file names to order numerically
        for index, name in enumerate(frame_names):
            frame_names[index] = name.replace("frame", "")

        # Order frame names numerically
        frame_names.sort(key=lambda line: int(line.split(".")[0]))

        if not resize:
            im = cv2.imread('{}/{}'.format(frame_folder_path, frame_names[0]))
            height, width, _ = im.shape
        else:
            width, height = ratios[0], ratios[1]

        for name in frame_names:
            frame_name = "{}".format(name)
            im = cv2.imread('{}/{}'.format(frame_folder_path, frame_name))
            resized_im = cv2.resize(im, (width, height))
            # Test if the keypoints are not NaN values
            if not np.nan in self.kps[0]:
                try:
                    self.draw_links_coloured(resized_im, self.kps[frame_names.index(name)], self.connections, [width, height])
                except IndexError as e:
                    print(e)

            cv2.imwrite("{}/{}".format(output_folder_path, name), resized_im)

        print("Processing Complete")
        return True



    def output_frames_to_video(self, frame_folder_path, output_video_path, output_video_name, fps, range=None):
        '''Overlay skeleton on original video.
        Args:
            frame_folder_path (Str) : Path to folder containing video frames.
            output_video_path (Str) : Path to store created video.
            output_video_name (Str) : Name given to the output video.
            fps (Int) : Fps at which to play back the created video.
        '''   
        frame_names = os.listdir(frame_folder_path)
        # Order frame names numerically
        frame_names.sort(key=lambda line: int(line.replace("Mirrored_", "").split(".")[0]))
        if range:
            frame_names[range[0]:range[1]]

        first_image = cv2.imread("{}/{}".format(frame_folder_path, frame_names[0]))
        height, width, _ = first_image.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f"{output_video_path}/{output_video_name}.avi", fourcc, fps, (width, height), True)

        for frame in frame_names:
            try:
                im = cv2.imread(f"{frame_folder_path}/{frame}")
                out.write(im)
            # If there are more frames than data points (missing data), exit loop and annotate up to that point
            except IndexError:
                break

        out.release()
        self.convert_avi_to_mp4(output_video_path, output_video_path.replace("avi", "mp4"))
        return True
    