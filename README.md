# PitchPerfectImage

## Project Description
There are multiple stages to this project as explained below;
1. **Video downloading:** From the provided list of videos for which we have to perform image extraction we first download the video from youtube making it easier to work with local video files.
2. **Frames extraction from Video:** The project uses `cv2` library to extract image from the video, the rate at which to capture images is configurable. `(.env file)`
3. **Extracting applicable images:** This step in the overall pipeline uses ViT model with prompts to identify most relavant images.
4. **Deduplicating images:** In this step we remove the duplicate images using the image encoder feature of Clip model and filter out duplicate images giving us a set of unique images.
5. **Image enahancement:** At this stage we enhance the image, enhance could be making the image sharper, resizing...etc. For now we do not use any Deep learning models and rely on a simple library function to achieve the results, but the same could be done with more sophisticated techniques using neural network.

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

## Models used for the project

### ViT-B/32
We use ViT-B/32 to select relavant images out of all the frames extracted from the video.

ViT-B/32 is a computer vision model designed for image classification (and potentially other vision tasks like object detection or segmentation, if fine-tuned appropriately). It takes an image as input, divides it into 32x32 pixel patches, processes these patches using a Transformer encoder, and outputs a prediction of what the image represents (e.g., "cat," "dog," "car"). It's a "Base" size model, making it a good balance between performance and computational requirements.

#### Description of VisionTransformation (ViT)
- Vision Transformer (ViT):

    Transformers for Images: ViT is a groundbreaking architecture that applies the Transformer model, originally designed for natural language processing (NLP), to the task of image classification (and other computer vision tasks). Traditional convolutional neural networks (CNNs) use convolutional layers to process images. ViTs, on the other hand, treat images as sequences of patches, similar to how Transformers treat sentences as sequences of words.

    How it Works (in brief):
    1. Image Patching: The input image is divided into fixed-size, non-overlapping patches (e.g., 16x16 pixels or 32x32 pixels).
    2. Linear Embedding: Each patch is flattened into a 1D vector and then linearly projected (embedded) into a higher-dimensional space. This creates a sequence of patch embeddings.
    3. Positional Encoding: Positional embeddings are added to the patch embeddings. Since Transformers are permutation-invariant (they don't inherently know the order of the input), positional embeddings provide information about the location of each patch within the original image. This is crucial for images, as spatial relationships are important.
    4. Transformer Encoder: The sequence of patch embeddings (with positional information) is fed into a standard Transformer encoder. The encoder consists of multiple layers of:
        Multi-Head Self-Attention: This is the core of the Transformer. It allows each patch embedding to "attend" to all other patch embeddings, capturing relationships between different parts of the image.
        Feed-Forward Network: A simple fully connected network applied to each patch embedding independently.
        1. Classification Head: A classification head (typically a simple multi-layer perceptron, MLP) is added on top of the Transformer encoder's output. This head takes the output of the encoder (usually the embedding corresponding to a special "[CLS]" token, similar to BERT) and produces a probability distribution over the possible image classes.

**NOTE: ViT-B/32, may not be the best model out there for image classfication, but a good starting point based on certain factors like the compute power available, accuracy...etc**

#### Some of the alternates to ViT available (and considered were ResNet)
- **Convolutional Neural Networks (CNNs):**
  - **ResNet:** A family of CNN models with residual connections, such as ResNet-18, ResNet-50, and ResNet-101.
  - **VGG:** A series of CNN models developed by the Visual Geometry Group, such as VGG-16 and VGG-19.
  - **InceptionNet:** A CNN model with an inception module, such as Inception-v3 and Inception-v4.

- **Hybrid Models:**
  - **DeiT (Data-efficient Image Transformer):** A Vision Transformer model that incorporates CNN-based features.
  - **PVT (Pyramid Vision Transformer):** A Transformer-based model that uses a pyramid structure to capture multi-scale features.
  - **Swin Transformer:** A Transformer-based model that uses a shifted window approach to capture local and global features.

### CLIP (openai/clip-vit-base-patch32)
We primarily use CLIP to identify similar images based on the score. We use the image encoder part of this model to arrive at a score and filter similar images giving us a set of unique images.

CLIP, developed by OpenAI, learns to understand the relationship between images and text descriptions. It doesn't just classify images into predefined categories; it learns a joint embedding space where images and their corresponding text descriptions are close together, and unrelated images and text are far apart.

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