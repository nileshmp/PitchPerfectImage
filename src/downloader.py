import yt_dlp as youtube_dl
import os
from logger_config import logger

class Downloader:
    def __init__(self, download_folder):
        #Initialize the Car with default attributes
        self.download_folder = download_folder
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
            logger.debug("Directory created successfully!")
        else:
            logger.debug("Directory already exists!")

        
    def download(self, video_url):
        """Downloads the video from YouTube."""
        logger.debug("downloading %s", video_url)
        ydl_opts = {
            # 'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 
            'outtmpl': f'{self.download_folder}/%(title)s/%(title)s.%(ext)s',  # Output filename
            'quiet': True,  # Suppress verbose output
            'no_warnings': True,
        }
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return 'temp_video.mp4'  # Return path to downloaded video
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None
        
if __name__=="__main__":
    Downloader("./data/videos")