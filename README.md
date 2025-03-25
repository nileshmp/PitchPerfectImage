# PitchPerfectImage

## Project Directory Structure
- `main.py`
  - Main file and entry point for the application to run. Its main function is to orchestrate the complete pipeline from loading ENVIRONMENT variables, dowloading youtube videos, extract relavant images and enhance extracted images.
- `poetry.toml`
  - Defines all the dependencies of the module.
- `.data/`
  - Directory where the youtube videos will be downloaded and the images created.
- `.env`
  - ENVIRONMENT configurations, details explanation given below.
- `src/config`
  - Module directory which stores varies configurations required for the program to run, currently has logging specific configuration.
- `src/file`
  - Module directory containing theutility methods pertaining to file/directory related operations.
- `src/image`
  - Module directory containing files which has functionality to enahncements, resize images. 
- `src/llm`
  - Module directory containing files containing functionality to execute LLM models and  prompt engineering.
- `src/parsers`
  - Module directory containing code to parse CSV, XLSX...
- `src/utils`
  - Module directory containing code to load ENVIRONMENT variables, downloading youtube videos.
- `src/examples`
  - Module directory containing sample example code to independently run experiments, before incorporating them into the main module. The idea is the ability to run each example and see the working.

## How to prepare environment, dependencies and run the project.
- The source code is availabel in public repository `https://github.com/nileshmp/PitchPerfectImage` 
- To clone the git repository run `git clone git@github.com:nileshmp/PitchPerfectImage.git`
- Install `ffmpeg` and `poetry`
- Post installing the required tools/libraries under directory `PitchPerfectImage (cd PitchPerfectImage)`  run `poetry install`. This will install project dependencies and prepare your project's virtual environment. (this project uses poetry for dependency management)
- Post the success of `poetry install` under directory `PitchPerfectImage` run the following command `poetry run python main.py` this will execute the programe. The execution does the following;\
  - read the excel file from location `data/Hindi Pitch Videos for Image extraction+enhancement.xlsx` and download all the youtube videos.
  - extract frames from the video under `data/videos` folder, extracts title from the youtube video and downloads videos under that.
  - run the frames through model with promt to pick the most relavant images (based on overall score)
  - Once scored we deduplicate the images.
  - and the final step is to enhance images.
- **All the artifacts (videos and images/frames) are created under different folder under ./data/videos/ for example:**
  
        ├── 20231121143501_C1290908_F12304_M5753166
            │   ├── 20231121143501_C1290908_F12304_M5753166.mp4
            │   └── frames
            │       ├── applicable-images
            │       │   ├── frame_0002.JPEG
            │       │   ├── frame_0010.JPEG
            │       │   ├── frame_0018.JPEG
            │       │   ├── frame_0027.JPEG
            │       │   ├── frame_0038.JPEG
            │       │   ├── frame_0043.JPEG
            │       │   ├── frame_0054.JPEG
            │       │   ├── frame_0056.JPEG
            │       │   ├── frame_0077.JPEG
            │       │   └── frame_0102.JPEG
            │       ├── dedup-images
            │       │   ├── frame_0002.JPEG
            │       │   ├── frame_0010.JPEG
            │       │   ├── frame_0027.JPEG
            │       │   ├── frame_0038.JPEG
            │       │   └── frame_0102.JPEG
            │       ├── enhanced-images
            │       │   ├── frame_0002.JPEG
            │       │   ├── frame_0010.JPEG
            │       │   ├── frame_0027.JPEG
            │       │   ├── frame_0038.JPEG
            │       │   └── frame_0102.JPEG
            │       ├── frame_0000.JPEG
            │       ├── frame_0001.JPEG
            │       ├── frame_0002.JPEG
            │       ├── frame_0003.JPEG

    and the first level folder looks like;

        data/videos
            ├── #SBIC#glasslampdecoration
            ├── 20231121143501_C1290908_F12304_M5753166
            ├── 3 December 2024
            ├── Accessory Geeks
            ├── Catalyst Crew
            ├── Crafted Crystels #jewellery #Bangles #handmade
            ├── December 3, 2024
            ├── Decor World team #decoration #items
            ├── Guardian Time
            ├── Nature thinker's
            ├── PENKRITI
            ├── SBIC SATHI HSS KACHNARI ＂पेपर क्राफ्ट्स＂
            ├── SBIC group
            ├── Sbic tejasvi karyakram।। विद्यार्थियों ने बनाया पर्स और पूजा की टोकरी
            ├── TechnoBots
            ├── The bouqet of Sayali
            ├── Trailblazer
            ├── tejasvi team sweksha
            ├── चुकंदर से लिप बाम कैसे बनाए।।
            └── पुरानी चुड़ियों, पुराने कपड़ों व ऊन से तैयार किए गए लटकन (side door hanging)

- There will be a log file created under the root directory `udhyam.log`

## Configurations
### .env
Contains the following configurations;

- SIMILARITY_MODEL_NAME=openai/clip-vit-base-patch32
- MODEL_NAME=ViT-B/32
- EXCLUSION_LIST=.DS_Store
- DOWNLOAD_FOLDER=./data/videos
- RAW_FRAMES_FOLDER=./data/videos/{}/frames
- APPLICABLE_FRAMES_FOLDER=/applicable-images
- DEDUP_FRAMES_FOLDER=/dedup-images
- ENHANCED_FRAMES_FOLDER=/enhanced-images
- FINAL_FRAMES_FOLDER=/final-images
- FRAME_INTERVAL_IN_SECONDS=1
- SIMILARITY_THRESHOLD=0.9
- #JPEG or WEBP
- IMAGE_FORMAT=JPEG 
- IMAGE_QUALITY=100
- TOP_RESULTS_COUNT=10

### log configurations
Directory `src/config` contains file `logger_config.py` which contains the logger configurations like;
    - setting logger level
    - setting log format
    - configuring different handlers like file_handler, console_output_handler...etc.

## TODO
- Move the configurations like log level to .env file (making it easier to set log level)
- Work on separating PROMT_ENGINEERING and making it more configurable and esier to change and experiment.