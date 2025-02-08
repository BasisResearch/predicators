from typing import List

from predicators.structs import Video

class VLMClassificationApproach:
    def __init__(self):
        self.vlm = ...
    
    def predict(self, support_videos: List[Video], 
                      support_labels: List[int], 
                      query_videos: List[Video]) -> List[int]:
        """Predict the labels for the query video.
        """
        return [0, 1]