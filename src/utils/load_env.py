from dotenv import load_dotenv
import os

load_dotenv()
# Get the database URL.  Provide a default for local development.
MODEL_NAME = os.getenv('MODEL_NAME')
DOWNLOAD_FOLDER = os.getenv('DOWNLOAD_FOLDER')
RAW_FRAMES_FOLDER = os.getenv('RAW_FRAMES_FOLDER')
APPLICABLE_FRAMES_FOLDER = os.getenv('APPLICABLE_FRAMES_FOLDER')
DEDUP_FRAMES_FOLDER = os.getenv('DEDUP_FRAMES_FOLDER')
ENHANCED_FRAMES_FOLDER = os.getenv('ENHANCED_FRAMES_FOLDER')
FINAL_FRAMES_FOLDER = os.getenv('FINAL_FRAMES_FOLDER')
EXCLUSION_LIST = os.getenv('EXCLUSION_LIST')
SIMILARITY_MODEL_NAME = os.getenv('SIMILARITY_MODEL_NAME')
FRAME_INTERVAL_IN_SECONDS = int(os.getenv('FRAME_INTERVAL_IN_SECONDS'))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD'))
IMAGE_QUALITY = int(os.getenv('IMAGE_QUALITY'))
IMAGE_FORMAT = os.getenv('IMAGE_FORMAT')
TOP_RESULTS_COUNT = int(os.getenv('TOP_RESULTS_COUNT'))

# Get an API key.  This is sensitive, so we don't provide a default.
#  If it's missing, we raise an error.
# api_key = os.getenv('API_KEY')
# if not api_key:
#     raise ValueError("API_KEY environment variable must be set.")

# # Get a port number, defaulting to 8000.
# port = int(os.getenv('PORT', 8000))

# # Get a boolean flag (e.g., for debug mode).
# debug_mode = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# print(f"Database URL: {database_url}")
# print(f"API Key: {api_key}")  # Careful with printing sensitive data!
# print(f"Port: {port}")
# print(f"Debug Mode: {debug_mode}")
