import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity  # For distinctiveness
from ..config.logger_config import logger
import clip
import torch
import shutil
from ..utils import load_env as ENV
from ..file.file_utils import FileUtils
from ..image.image_enhancer import ImageUtil
from .vit import ViT


class Clip:
    def __init__(self, model_name, use_clip=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.use_clip = use_clip  # Flag to use CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(ENV.MODEL_NAME, device=device)
        self.clip_model, self.clip_preprocess = self._load_model(model_name)
        self.fileUtils = FileUtils()
        self.imageUtil = ImageUtil()
        self.vit = ViT(device)

    def _load_model(self, model_name):
        if self.use_clip:
            logger.debug("Inside using clip.")
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained(model_name).to(self.device)
            preprocess = CLIPProcessor.from_pretrained(model_name)
            logger.debug("Using model %s", model_name)
            return model, preprocess
        else:
            logger.debug("using resenet50 pretrained model")
            model = models.resnet50(pretrained=True).to(self.device)
            model.eval()  # Set to evaluation mode
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return model, preprocess

    def _get_frame_embedding(self, count, frame, prompts):
        """Gets the CLIP embedding for a single frame."""
        if self.use_clip:
            if len(prompts) == 0:
                prompts = ""
            logger.debug(prompts)
            inputs = self.clip_preprocess(text="", images=frame, return_tensors="pt", padding=True)
            # Handle different possible shapes (Scenario 3 is most likely after unsqueeze)
            if inputs['pixel_values'].ndim == 3:  # (C, H, W) - already handled by unsqueeze
                input_tensor = inputs["pixel_values"].unsqueeze(0).to(self.device)
            elif inputs['pixel_values'].ndim == 4 and inputs['pixel_values'].shape[1] == 3: # (B, C, H, W) - already has batch dim
                input_tensor = inputs["pixel_values"].to(self.device)
            elif inputs['pixel_values'].ndim == 3:  # (H, W, C) - permute!
                input_tensor = inputs["pixel_values"].permute(2, 0, 1).unsqueeze(0).to(self.device)
            elif inputs['pixel_values'].ndim == 4: # (B, H, W, C) - permute!
                input_tensor = inputs["pixel_values"].permute(0, 3, 1, 2).to(self.device)

            else:
                raise ValueError(f"Unexpected shape for pixel_values: {inputs['pixel_values'].shape}")
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(input_tensor)
            return embedding.squeeze(0).cpu().numpy()
        else:
            #handle no clip case
            return None
    
        
    def _select_representative_frames(self, embeddings, threshold=0.9):
        """Selects representative frames based on cosine similarity."""
        selected_indices = []
        embedding_count = 0;
        for score, image_name, image_path, embedding in embeddings:
            if embedding_count == 0:
                selected_indices.append((score, image_name, image_path, embedding))
                embedding_count += 1
                continue
            is_distinct = True
            for j, value in enumerate(selected_indices):
                similarity = cosine_similarity(embeddings[embedding_count][3].reshape(1, -1), embeddings[j][3].reshape(1, -1))[0][0]
                logger.debug(f"Similarity score found to be {similarity}")
                if similarity > threshold:
                    is_distinct = False
                    break
            if is_distinct:
                selected_indices.append((score, image_name, image_path, embedding_count))
            logger.debug(f"Entered for index {embedding_count}")
            embedding_count += 1
        return selected_indices

    def process_video(self, video_path, frame_save_folder, prompts, frame_interval=5, similarity_threshold=0.9):
        """Main function to process the video."""
        frames = self.imageUtil.extract_frames(video_path, frame_interval)
        # we use the ViT model and extract images corresponding to the prompts
        applicable_frames =  self.vit.get_applicable_images(frames, frame_save_folder, prompts)
        logger.debug("Count of Frames is %d", len(frames))
        applicable_frames_folder = frame_save_folder + ENV.APPLICABLE_FRAMES_FOLDER
        self.fileUtils.creat_if_not_exists(applicable_frames_folder)
        top_results_count = ENV.TOP_RESULTS_COUNT
        logger.debug(f"Top {top_results_count} Representative Frames:")
        embeddings = []
        # we use openai/clip-vit-base-patch32 to extract the embeddings of an image to filter out similar
        # images, finally ending up with a unique set of images representation.
        for frame_number, score, image, image_name, image_path in applicable_frames[:top_results_count]:
            print(f"- Frame {frame_number}: Score = {score:.4f}, Path = {image_name}")
            image.save(applicable_frames_folder + f"/{image_name}", ENV.IMAGE_FORMAT, quality=ENV.IMAGE_QUALITY)
            embeddings.append((score, image_name, image_path, self._get_frame_embedding(frame_number, image, prompts)))
        logger.debug(f"Count of embeddings {len(embeddings)}")
        # logger.debug(f"Embeddings are : \n{embeddings}")
        representative_indices = self._select_representative_frames(embeddings, similarity_threshold)
        logger.debug(f"Count of representative indices is {len(representative_indices)}")
        # logger.debug(f"Representative indices are {representative_indices}")
        dedup_frames_folder = frame_save_folder + ENV.DEDUP_FRAMES_FOLDER
        self.fileUtils.creat_if_not_exists(dedup_frames_folder)
        enhanced_frames_folder = frame_save_folder + ENV.ENHANCED_FRAMES_FOLDER
        self.fileUtils.creat_if_not_exists(enhanced_frames_folder)
        for score, image_name, image_path, embedding in representative_indices:
            shutil.copy2(image_path, dedup_frames_folder)
            image = cv2.imread(image_path)
            enhanced_image = self.imageUtil.enhance_resolution(image)
            cv2.imwrite(f"{enhanced_frames_folder}/{image_name}", enhanced_image)