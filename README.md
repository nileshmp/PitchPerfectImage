# PitchPerfectImage

## How to prepare environment, dependencies and run the project.
- The source code is availabel in public repository `https://github.com/nileshmp/PitchPerfectImage` 
- To clone the git repository run `git clone git@github.com:nileshmp/PitchPerfectImage.git`
- Install `ffmpeg` and `poetry`
- Post installing the required tools/libraries under directory `PitchPerfectImage (cd PitchPerfectImage)`  run `poetry install`. This will install project dependencies and prepare your project's virtual environment. (this project uses poetry for dependency management)
- Post the success of `poetry install` under directory `PitchPerfectImage` run the following command `poetry run python main.py` this will execute the programe. The execution does the following;\
  - read the excel file and download all the youtube videos.
  - extract frames from the video
  - run the frames through model with promt to pick the most relavant images (based on overall score)
  - Once scored we deduplicate the images.
  - and the final step is to enhance images.