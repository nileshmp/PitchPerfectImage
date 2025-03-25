import logging
import sys


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s",  # Log format
    datefmt="%d-%m-%Y %H:%M:%S" 
)

# Create a logger instance
logger = logging.getLogger(__name__)  # Root logger

stdoutHandler = logging.StreamHandler(stream=sys.stdout)
fileHandler = logging.FileHandler("udhyam.log")

# Set the log levels on the handlers
stdoutHandler.setLevel(logging.DEBUG)
fileHandler.setLevel(logging.DEBUG)

# Create a log format using Log Record attributes
# fmt = logging.Formatter(
    # "%(asctime)s - %(name)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s"
# )

# Set the log format on each handler
# stdoutHandler.setFormatter(fmt)
# fileHandler.setFormatter(fmt)

fmt = logging.Formatter(
    "%(asctime)s - %(name)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S" 
)

# Set the log format on each handler
stdoutHandler.setFormatter(fmt)
fileHandler.setFormatter(fmt)

# Add each handler to the Logger object
logger.addHandler(stdoutHandler)
logger.addHandler(fileHandler)