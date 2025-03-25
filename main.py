from src.llm.processor import VideoProcessor
from src.parsers.parser import XLSX
import os
from src.config.logger_config import logger
import src.utils.load_env as ENV
from src.file.file_utils import FileUtils
from src.utils.downloader import Downloader

class Main:

    def __init__(self):
        self.fileUtils = FileUtils()

    def parse_csv(self, file_path):
        xlsx = XLSX()
        return xlsx.parse(file_path)

    def dowload_videos(self, videos):
        self.fileUtils.creat_if_not_exists(ENV.DOWNLOAD_FOLDER)
        downloader = Downloader(ENV.DOWNLOAD_FOLDER)
        for video in videos:
            logger.debug(video)
            downloader.download(video)

    def process_videos(self, prompts):
        logger.debug("Inside process video method")
        # traverse folder, extract videos folder name and file
        video_processor = VideoProcessor(model_name=ENV.SIMILARITY_MODEL_NAME, use_clip=True)
        dir_files_dict = self.fileUtils.walk_files(ENV.DOWNLOAD_FOLDER)  
        logger.debug(f"Dictionary of dir and files are : \n {dir_files_dict}")
        for dir in dir_files_dict:
            video_file_path = dir_files_dict[dir]
            frame_save_folder = ENV.RAW_FRAMES_FOLDER.format(self.fileUtils.base_dir(dir))
            video_processor.process_video(video_path=video_file_path, frame_save_folder=frame_save_folder, prompts=prompts, frame_interval=ENV.FRAME_INTERVAL_IN_SECONDS, similarity_threshold=ENV.SIMILARITY_THRESHOLD)

if __name__ == "__main__":
    logger.debug(f"Printin all the environment variables: ")
    for key, value in os.environ.items():
        logger.debug(f"{key}: {value}")
    
    # prompts = [
    #     "A photo of a person with their eyes open.",
    #     "A photo of a person with their eyes closed.",
    #     # "A photo of a person with their eyes open, in a still pose.",
    #     # "A clear photo of a person looking at the camera, not moving.",
    #     # "A clear photo of a person looking at the camera, with complete face visible.",
    #     # "A person with open eyes, standing still.",
    #     # "A clear image of the object.", 
    #     # "A clear image of the object with proper lighting ."
    # ]

    # prompts = [
    #     "A visually clear image that best represents a startup business idea, focusing on the product, branding, or prototype.",
    #     "A frame that clearly shows a product, prototype, or demonstration of a business idea in an engaging and understandable manner.",
    #     "A sharp, high-quality image that is well-lit, visually clear, and aesthetically appealing, making the business idea easily understandable."
    # ]
    prompts = {
            "business_idea": [
                "A clear depiction of the core business idea.",
                "An image representing the startup's product or service.",
                "A visual explanation of the company's value proposition.",
                "The main concept of the business is shown."
            ],
            "branding": [
                "The company logo is visible.",
                "The brand's colors and visual style are present.",
                "An image showcasing the company's branding elements.",
                "Strong branding elements are displayed."
            ],
            "team_representation": [
                "The founding team is shown.",
                "A picture of the team members working together.",
                "A diverse and inclusive team is represented.",  # Important for fairness!
                "Team members are present in the image."
            ],
            "clarity_appeal": [
                "A high-quality, visually appealing image.",
                "A clear and easy-to-understand visual.",
                "The image is well-lit and in focus.",
                "A professional and polished visual presentation."
            ]
        }
    main = Main()
    videos_link = main.parse_csv("./data/Hindi Pitch Videos for Image extraction+enhancement.xlsx")
    # main.dowload_videos(videos_link)
    main.process_videos(prompts)

    # print(f"Exclusion list is : {os.getenv("EXCLUSION_LIST")}")
    # print(f"Download folder is : {os.getenv("./data/videos")}")
    # videos_link = main.parse_csv("/Users/nilesh/work/Aikyam/clients/Udhyam/assignment/Hindi Pitch Videos for Image extraction+enhancement.xlsx")
    # main.dowload_videos(videos_link, "./data/videos")
    # main.process_videos("./data/videos", prompts)
    # process the images for repeat (code already there)
    # Improve image quality through noise reduction, resolution enhancement, and brightness/contrast/colour adjustments.
    # Optionally, apply minor corrections (e.g., cropping, background enhancement) to ensure clarity and focus.
    # Ensure the automated image processing is optimized for display on the PWA.

    # main.walk_files("./data/videos")
