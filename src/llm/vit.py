import torch
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity  # For distinctiveness
from ..config.logger_config import logger
import clip
import torch
from ..utils import load_env as ENV
from ..file.file_utils import FileUtils
from ..image.image_enhancer import ImageUtil

class ViT:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load(ENV.MODEL_NAME, device=device)
        self.fileUtils = FileUtils()
        self.imageUtil = ImageUtil()

    def get_applicable_images(self, frames, save_folder, prompts):
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