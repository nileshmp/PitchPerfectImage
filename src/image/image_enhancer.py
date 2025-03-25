import cv2
import numpy as np

from ..config.logger_config import logger

class ImageUtil:

    def gaussian_blurring(self, image):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)  # (5, 5) is the kernel size, 0 is sigmaX (standard deviation)

        cv2.imshow('Original', image)
        cv2.imshow('Blurred', blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def median_blurring(self, image):
        median_blurred = cv2.medianBlur(image, 5)  # 5 is the kernel size (must be odd)
        cv2.imshow('Original', image)
        cv2.imshow('Median Blurred', median_blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def bilateral_filtering(self, image):
        bilateral_filtered = cv2.bilateralFilter(image, 9, 75, 75)  # 9 is diameter, 75 is sigmaColor, 75 is sigmaSpace
        cv2.imshow('Original', image)
        cv2.imshow('Bilateral Filtered', bilateral_filtered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def denoising(self, image):
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised

    def enhance_resolution(self, image):
        blurred = cv2.GaussianBlur(image, (0, 0), 3)  # Adjust sigma (3 here) for blur strength
        # 2. Create the unsharp mask
        unsharp_mask = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)  # 1.5 and -0.5 are weights, 0 is gamma
        return unsharp_mask
        
    
    def laplace_sharpening(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale

        # Apply Laplacian operator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Convert back to uint8 (absolute value and scaling)
        laplacian = np.uint8(np.absolute(laplacian))

        # sharpen the image by adding the laplacian to the original
        sharpened = cv2.addWeighted(gray, 1, laplacian, 0.7, 0)
        #convert back to color if original image was color
        sharpened_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        return sharpened_color

    def contract_colour_enhancer(self, image):
        # Apply histogram equalization
        equalized = cv2.equalizeHist(image)
        return equalized

    def clahe(self, image):
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Apply CLAHE
        clahe_equalized = clahe.apply(image)
        return clahe_equalized

    def gamma_correction(self, image):
        gamma = 1.5  # Adjust this value (1.0 = no change, < 1.0 brightens, > 1.0 darkens)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(image, table)
        return gamma_corrected

    def color_adjester(self, image):
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Increase saturation (adjust the factor as needed)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.3
        hsv[:,:,1] = np.clip(hsv[:,:,1],0,255) #clip values to stay between 0-255

        # Increase value (brightness)
        hsv[:, :, 2] = hsv[:, :, 2] * 1.1
        hsv[:,:,2] = np.clip(hsv[:,:,2],0,255)


        # Convert back to BGR
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return enhanced

    def resize(self, image):
        height, width = image.shape[:2]
        resized_image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        return resized_image

    def super_resolution(self, image):
        model = cv2.dnn_superres.DnnSuperResImpl_create()
        model.readModel("EDSR_x4.pb")  # Make sure to download the model
        model.setModel("edsr", 4)  # EDSR model with a scale factor of 4

        # Perform super-resolution
        super_res_image = model.upsample(image)
        return super_res_image
    
    def extract_frames(self, video_path, frame_interval=5):
        """Extracts frames from the video at a given interval (in seconds)."""
        logger.debug("Inside extract frames method.")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        extracted_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % int(fps * frame_interval) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted_frames.append(frame)
            frame_count += 1

        cap.release()
        logger.debug("Length of the extracted frames is %d", len(extracted_frames))
        logger.debug("Length of the frame count is %d", frame_count)
        return extracted_frames


if __name__ == "__main__":
    image = cv2.imread('/Users/nilesh/work/Aikyam/clients/Udhyam/assignment/PitchPerfectImage/data/videos/20231121143501_C1290908_F12304_M5753166/frames/dedup-images/frame_0027.JPEG')
    image_util = ImageUtil()
    # image_util.gaussian_blurring(image)
    # image_util.median_blurring(image)
    # image_util.bilateral_filtering(image)
    # image_util.denoising(image)
    # image_util.enhance_resolution(image) # use this processing
    # image_util.laplace_sharpening(image)
    # image_util.contract_colour_enhancer(image) # have to fix this method, giving exception
    # image_util.clahe(image) # have to fix this method, giving exception
    # image_util.gamma_correction(image)
    # image_util.color_adjester(image)
    # image_util.resize(image)
    enahnced_image = image_util.super_resolution(image)
    cv2.imshow('Original', image)
    cv2.imshow('Enhanced', enahnced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


