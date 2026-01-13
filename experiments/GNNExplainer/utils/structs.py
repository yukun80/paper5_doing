"""
Module: structs.py
Location: experiments/GNNExplainer/utils/structs.py
Description:
    Defines shared data structures for the GNNExplainer module.
    Ensures type consistency and pickling compatibility between the 
    Explanation Engine (explain_landslide.py) and Visualization Tools.

Author: AI Assistant (Virgo Edition)
Date: 2026-01-12
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np

@dataclass
class ExplanationArtifact:
    """
    A portable container for a single node's explanation result.
    
    Attributes:
        su_id (str): The Slope Unit identifier.
        node_idx (int): The internal graph node index.
        dataset_split (str): 'train', 'test', or 'val'.
        prediction_prob (float): The model's predicted probability for class 1 (Landslide).
        true_label (int): The ground truth label (0 or 1).
        neighbor_indices (List[str]): SU_IDs of the involved neighbors (subgraph).
        edge_weights (Dict[Tuple[str, str], float]): Importance weights for edges in the subgraph.
                                                     Key is (source_su, target_su).
        feature_mask (np.ndarray): The learned importance mask for node features.
        feature_names (List[str]): Names corresponding to the feature mask.
        node_attributes (Dict[str, Dict[str, float]]): Key attribute values for the center and neighbor nodes.
                                                       Format: {su_id: {'slope': val, 'dNDVI': val...}}
    """
    su_id: str
    node_idx: int
    dataset_split: str
    prediction_prob: float
    true_label: int
    neighbor_indices: List[str]
    edge_weights: Dict[Any, float]
    feature_mask: np.ndarray
    feature_names: List[str]
    node_attributes: Dict[str, Dict[str, float]] = field(default_factory=dict)
