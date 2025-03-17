from src import processor, downloader
from src import parser
import os
from logger_config import logger

class Main:

    def parse_csv(self, file_path):
        xlsx = parser.XLSX()
        return xlsx.parse(file_path)

    def dowload_videos(self, videos, download_folder):
        downloader = downloader.Downloader(download_folder)
        for video in videos:
            logger.debug(video)
            downloader.download(video)

    def process_videos(self, download_folder):
        logger.debug("Inside process video method")
        prompts = [
            "A photo of a person with their eyes open.",
            "A photo of a person with their eyes closed.",
            # "A photo of a person with their eyes open, in a still pose.",
            # "A clear photo of a person looking at the camera, not moving.",
            # "A clear photo of a person looking at the camera, with complete face visible.",
            # "A person with open eyes, standing still.",
            # "A clear image of the object.", 
            # "A clear image of the object with proper lighting ."
        ]

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
        model_name = "openai/clip-vit-base-patch32"
        # traverse folder, extract videos folder name and file
        video_processor = processor.VideoProcessor(model_name=model_name, use_clip=True)
        dir_files_dict = self.walk_files(download_folder)  
        logger.debug(f"Dictionary of dir and files are : \n {dir_files_dict}")
        for dir in dir_files_dict:
            frame_save_folder = dir + "/frames"
            video_path = dir_files_dict[dir]
            video_processor.process_video(video_path=video_path, frame_save_folder=frame_save_folder, prompts=prompts, frame_interval=1, similarity_threshold=0.9)
        # def process_video(self, video_path, frame_save_folder, frame_interval=5, similarity_threshold=0.9):
        # processor.process_video( 1, 0.9)
    
    def walk_files(self, src_filepath = "."):
        exclusion_list = ['.DS_Store']
        filepath_list = {}
    
        #This for loop uses the os.walk() function to walk through the files and directories
        #and records the filepaths of the files to a list
        for root, dirs, files in os.walk(src_filepath):
            
            #iterate through the files currently obtained by os.walk() and
            #create the filepath string for that file and add it to the filepath_list list
            for file in files:
                if file in exclusion_list:
                    continue
                #Checks to see if the root is '.' and changes it to the correct current
                #working directory by calling os.getcwd(). Otherwise root_path will just be the root variable value.
                if root == src_filepath:
                    root_path = os.getcwd() + "/"
                else:
                    root_path = root
                
                #This if statement checks to see if an extra '/' character is needed to append 
                #to the filepath or not
                if (root_path != src_filepath):
                    filepath_list[root_path] = root_path + "/" + file
                else:
                    filepath_list[root_path] = root_path + file
                
        return filepath_list

if __name__ == "__main__":
    main = Main()
    # videos_link = main.parse_csv("/Users/nilesh/work/Aikyam/clients/Udhyam/assignment/Hindi Pitch Videos for Image extraction+enhancement.xlsx")
    # main.dowload_videos(videos_link, "./data/videos")
    main.process_videos("./data/videos")
    # process the images for repeat (code already there)
    # Improve image quality through noise reduction, resolution enhancement, and brightness/contrast/colour adjustments.
    # Optionally, apply minor corrections (e.g., cropping, background enhancement) to ensure clarity and focus.
    # Ensure the automated image processing is optimized for display on the PWA.

    # main.walk_files("./data/videos")
