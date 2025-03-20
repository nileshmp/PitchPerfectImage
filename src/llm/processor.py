import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity  # For distinctiveness
from ..config.logger_config import logger
import clip
import torch
import shutil
from ..utils import load_env as ENV
from ..file.file_utils import FileUtils


class VideoProcessor:
    def __init__(self, model_name, use_clip=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.use_clip = use_clip  # Flag to use CLIP model
        # self.device = device
        # self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(ENV.MODEL_NAME, device=device)
        self.clip_model, self.clip_preprocess = self._load_model(model_name)
        self.fileUtils = FileUtils()

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

    def _extract_frames(self, video_path, frame_interval=5):
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

    def _get_frame_embedding(self, count, frame, prompts):
        """Gets the CLIP embedding for a single frame."""
        if self.use_clip:
            if len(prompts) == 0:
                prompts = ""
            logger.debug(prompts)
            inputs = self.clip_preprocess(text="", images=frame, return_tensors="pt", padding=True)
            # self.print_score_4_prompts(count, inputs, prompts)
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
        # if len(embeddings) > 0:
        #     selected_indices.append(0)  # Always include the first frame
        embedding_count = 0;
        for score, image_name, image_path, embedding in embeddings:
            # logger.debug(f"Printing contents of embeddings {embeddings[embedding_count][3]}")
            if embedding_count == 0:
                selected_indices.append((score, image_name, image_path, embedding))
                embedding_count += 1
                continue
        # for i in range(1, len(embeddings)):
            is_distinct = True
            for j, value in enumerate(selected_indices):
                similarity = cosine_similarity(embeddings[embedding_count][3].reshape(1, -1), embeddings[j][3].reshape(1, -1))[0][0]
                logger.debug(f"Similarity score found to be {similarity}")
                if similarity > threshold:
                    is_distinct = False
                    break
            if is_distinct:
                selected_indices.append((score, image_name, image_path, embedding_count))
            # selected_indices.append(i)
            logger.debug(f"Entered for index {embedding_count}")
            embedding_count += 1
        return selected_indices

    def process_video(self, video_path, frame_save_folder, prompts, frame_interval=5, similarity_threshold=0.9):
        """Main function to process the video."""
        frames = self._extract_frames(video_path, frame_interval)
        applicable_frames =  self.get_scores_from_frames(frames, frame_save_folder, prompts)
        logger.debug("Count of Frames is %d", len(frames))
        applicable_frames_folder = frame_save_folder + ENV.APPLICABLE_FRAMES_FOLDER
        self.fileUtils.creat_if_not_exists(applicable_frames_folder)
        top_results_count = ENV.TOP_RESULTS_COUNT
        logger.debug(f"Top {top_results_count} Representative Frames:")
        embeddings = []
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
        for score, image_name, image_path, embedding in representative_indices:
            shutil.copy2(image_path, dedup_frames_folder)

    def get_scores_from_frames(self, frames, save_folder, prompts):
        frame_count = 0
        results = []
        self.fileUtils.creat_if_not_exists(save_folder)
        for i, frame in enumerate(frames):
            frame_name = f"frame_{i:04d}.{ENV.IMAGE_FORMAT}"
            frame_path = os.path.join(save_folder, frame_name)
            # cv2.imwrite(frame_path, frame)
            frame_image = Image.fromarray(frame)
            frame_image.save(frame_path, ENV.IMAGE_FORMAT, quality=ENV.IMAGE_QUALITY)
            
            # logger.debug(f"Saved frame {i} to {frame_path}")
            all_scores = {}
            for category, prompt_list in prompts.items():
                scores = self.get_clip_score(frame_image, frame_path, prompt_list)
                all_scores[category] = scores

            # --- Calculate Total Score (Weighted or Simple Average) ---
            #  Here's a simple average; you can add weights if some criteria are more important.
            total_score = 0
            num_prompts = 0
            for category_scores in all_scores.values():
                total_score += sum(category_scores)
                num_prompts += len(category_scores)
            total_score /= num_prompts

            results.append((frame_count, total_score, frame_image, frame_name, frame_path))
            frame_count += 1

        # Sort results by total score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results            

    def get_clip_score(self, frame_image, frame_path, text_prompts):
        """
        Calculates CLIP scores for an image against a list of text prompts.

        Args:
            frame_image: Path to the image file.
            text_prompts: A list of text prompts.

        Returns:
            A list of similarity scores (one for each prompt).
        """
        image = Image.open(frame_path)
        # image = frame_image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in text_prompts]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            scores = similarity[0].cpu().numpy()

        return scores.tolist()