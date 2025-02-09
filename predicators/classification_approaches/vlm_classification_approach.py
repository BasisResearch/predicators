from typing import List, Tuple, Optional
import os
import re

import PIL
import logging

from predicators.structs import Video
from predicators import utils
from predicators.settings import CFG

class VLMClassificationApproach:
    def __init__(self):
        self._vlm = utils.create_vlm_by_name(CFG.vlm_model_name)
        self._max_video_len = 10
        self.log_dir: Optional[str] = None
    
    @classmethod
    def approach_name(cls) -> str:
        return "vlm_classification"
    
    def predict(self, support_videos: List[Video], 
                      support_labels: List[int], 
                      query_videos: List[Video],
                      task_id: int) -> List[int]:
        """Predict the labels for the query video.

        Args:
            support_videos (List[Video]): The support videos.
            support_labels (List[int]): The support labels.
            query_videos (List[Video]): The query videos.
            task_id (int): The task id, used in logging dir.
        """
        # --- Prepare the log directory ---
        self.log_dir = os.path.join(CFG.log_dir, self.approach_name(), 
                                    f"seed{CFG.seed}", f"task{task_id}")
        # remove the dir if it exists
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)

        # --- Prepare the prompt and images ---
        support_videos, query_videos = self._preprocess_videos(
                                                        support_videos, 
                                                        query_videos)
        prompt, imgs = self._prepare_prompt(support_videos, support_labels, 
                                            query_videos)

        # --- Get the response from the VLM ---
        response = self._vlm.sample_completions(prompt, imgs,
                                                CFG.vlm_temperature, 
                                                CFG.seed)[0]
        answer = self._save_and_parse_vlm_response(response)

        if answer["matching_video"] == "query_1":
            clf_answer = [1, 0]
        else:
            clf_answer = [0, 1]
        assert len(clf_answer) == len(query_videos), "Answer length mismatch."
        return clf_answer
    
    def _prepare_prompt(self, support_videos: List[Video],
                                support_labels: List[int],
                                query_videos: List[Video],
                            ) -> Tuple[str, List[PIL.Image.Image]]:
        """Prepare the prompt for the VLM by:
        1. Load the prompt from file
        2. add labels to the videos
        """
        del support_labels

        # --- Prepare the prompt ---
        prompt_path = os.path.join("prompts", "classification.outline")
        with open(prompt_path, "r") as f:
            prompt = f.read()
        
        # --- Prepare the images ---
        # Create a directory to save the images.
        imgs_dir = os.path.join(self.log_dir, "imgs")

        assert len(support_videos) == 1, "Currently assume only 1 support video."
        # Save for later inspection
        support_videos = [utils.add_label_to_video(video, 
                                prefix="ref_", 
                                imgs_dir=imgs_dir, save=True) for video 
                            in support_videos]
        query_videos = [utils.add_label_to_video(video, 
                                prefix=f"query{i}_", 
                                imgs_dir=imgs_dir, save=True) for i, video 
                            in enumerate(query_videos)]
        imgs = [img for video in support_videos + query_videos for img in video]

        return prompt, imgs

    def _save_and_parse_vlm_response(self, response_text: str) -> dict:
        """
        Parses the VLM response to extract the matching video choice.

        Args:
            response_text (str): The raw response from the VLM.

        Returns:
            dict: A dictionary with "matching_video" as 'query_1' or 'query_2',
                and "reasoning" as a string explaining the choice.
        """
        # --- Save the response ---
        response_path = os.path.join(self.log_dir, "response.txt")
        os.makedirs(os.path.dirname(response_path), exist_ok=True)
        with open(response_path, "w") as f:
            f.write(response_text)

        # --- Parse the response ---
        match_video = re.search(r"%% Matching Video:\s*(query_1|query_2)", 
                                response_text)
        match_reasoning = re.search(
                            r"%% Reasoning:\s*(.*?)(?=\n%% Matching Video:|$)", 
                            response_text, re.DOTALL)

        if match_video:
            matching_video = match_video.group(1)
            reasoning = match_reasoning.group(1).strip() if match_reasoning\
                            else "No reasoning provided."
            return {"matching_video": matching_video, "reasoning": reasoning}
        else:
            raise ValueError("Could not parse the response correctly.")

    def _preprocess_videos(self, support_videos: List[Video], 
                                query_videos: List[Video]
                                ) -> Tuple[List[Video], List[Video]]:
        """Preprocess the support and query videos.
        Subsample the videos to the max_video_len.
        """
        # Subsample the frames of the videos to the max_video_len.
        def subsample_video(video: Video) -> Video:
            if len(video) <= self._max_video_len:
                return video
            step = len(video) / self._max_video_len
            return [video[int(i * step)] for i in range(self._max_video_len)]

        support_videos = [subsample_video(video) for video in support_videos]
        query_videos = [subsample_video(video) for video in query_videos]

        return support_videos, query_videos