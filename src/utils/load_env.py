from dotenv import load_dotenv
import os

load_dotenv()
# Get the database URL.  Provide a default for local development.
MODEL_NAME = os.environ.get('MODEL_NAME')
DOWNLOAD_FOLDER = os.environ.get('DOWNLOAD_FOLDER')
EXCLUSION_LIST = os.environ.get('EXCLUSION_LIST')
SIMILARITY_MODEL_NAME = os.environ.get('SIMILARITY_MODEL_NAME')

# Get an API key.  This is sensitive, so we don't provide a default.
#  If it's missing, we raise an error.
# api_key = os.environ.get('API_KEY')
# if not api_key:
#     raise ValueError("API_KEY environment variable must be set.")

# # Get a port number, defaulting to 8000.
# port = int(os.environ.get('PORT', 8000))

# # Get a boolean flag (e.g., for debug mode).
# debug_mode = os.environ.get('DEBUG_MODE', 'False').lower() == 'true'

# print(f"Database URL: {database_url}")
# print(f"API Key: {api_key}")  # Careful with printing sensitive data!
# print(f"Port: {port}")
# print(f"Debug Mode: {debug_mode}")
