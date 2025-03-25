# PitchPerfectImage

## How to prepare environment, dependencies and run the project.
- The source code is availabel in public repository `https://github.com/nileshmp/PitchPerfectImage` 
- To clone the git repository run `git clone git@github.com:nileshmp/PitchPerfectImage.git`
- Install `ffmpeg` and `poetry`
- Post installing the required tools/libraries under directory `PitchPerfectImage (cd PitchPerfectImage)`  run `poetry install`. This will install project dependencies and prepare your project's virtual environment. (this project uses poetry for dependency management)
- Post the success of `poetry install` under directory `PitchPerfectImage` run the following command `poetry run python main.py` this will execute the programe. The execution does the following;\
  - read the excel file and download all the youtube videos.
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