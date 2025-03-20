import os
from src.config.logger_config import logger
import src.utils.load_env as ENV

class FileUtils:

    def base_dir(self, dir):
        last_directory = os.path.basename(dir)
        return last_directory

    def creat_if_not_exists(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)  # Create the output directory if it doesn't exist
            logger.debug(f"Created folder '{folder}'")
    
    def walk_files(self, src_filepath = "."):
        exclusion_list = [ENV.EXCLUSION_LIST]
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