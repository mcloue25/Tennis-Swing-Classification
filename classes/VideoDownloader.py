import os 

from tqdm import tqdm
from yt_dlp import YoutubeDL


def create_folder(folder_name: str):
    '''Used to create folders
    '''
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)



class YouTubeDownloader:
    ''' Class for downloading videos from URLs (YouTube) links
    '''
    def __init__(self, url_list, output_path):
        self.url_list = url_list
        self.output_path = output_path
        # Downloads best quality possible and saves to output folder
        self.ydl_opts = {
            "format": "bv*[ext=mp4]",
            "outtmpl": f"{self.output_path}" + "%(title)s.%(ext)s"
        }

    def download_videos(self):
        with YoutubeDL(self.ydl_opts) as ydl:
            for url in tqdm(self.url_list, desc="Downloading videos"):
                print(f"Downloading: {url}")
                ydl.download([url])
                print("-" * 30)


if __name__ == "__main__":
    url_list = [
        "https://www.youtube.com/watch?v=C4Gl-T2dtss",
        "https://www.youtube.com/watch?v=qXtJDJ1U7_8",
        "https://www.youtube.com/watch?v=CcvhT8kF5_Y",
        "https://www.youtube.com/watch?v=bt6dzSemFNM"
    ]
    output_path = 'data/videos/full_videos/'
    downloader = YouTubeDownloader(url_list, output_path)
    create_folder('data/videos/')
    create_folder(output_path)
    downloader.download_videos()
