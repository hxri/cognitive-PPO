import torch

# def anxiety(emotion):
#     F,A,J,S,D,U = emotion

#     anxiety_level = F * 0.35 + A * 0.25 + J * 0.10 + S * 0.15 + D * 0.10 + U * 0.05

#     return anxiety_level

def anxiety(emotion):
    """
    Maps emotion values to state and trait anxiety levels using the ERRM.

    Args:
        emotion_values (dict): A dictionary containing emotion values with emotion names as keys.
                               Example: {'fear': 0.8, 'anger': 0.4, 'joy': 0.2, 'sadness': 0.6, 'disgust': 0.1, 'surprise': 0.3}
        weights (dict): A dictionary containing weights for each emotion with emotion names as keys.
                        Example: {'fear': 0.5, 'anger': 0.3, 'joy': 0.2, 'sadness': 0.4, 'disgust': 0.1, 'surprise': 0.2}
        emotion_regulation (float): Emotion regulation level ranging from 0 to 1.

    Returns:
        tuple: A tuple containing state anxiety level and trait anxiety level, both in the range of 20 to 80.

    """
    # Calculate emotional reactivity score
    F,A,J,S,D,U = emotion

    anxiety_level = F * 0.35 + A * 0.25 + J * 0.10 + S * 0.15 + D * 0.10 + U * 0.05

    # # Determine state anxiety level
    # state_anxiety = 20 + (60 * anxiety_level * (1 - 0.3))  # Adjust with emotion regulation level

    # # Determine trait anxiety level
    # trait_anxiety = 20 + (60 * anxiety_level * (1 - 0.3) * 0.5)  # Adjust with emotion regulation level

    state_anxiety = anxiety_level * (1 - 0.5)  # Adjust with emotion regulation level

    # Determine trait anxiety level
    trait_anxiety = anxiety_level * (1 - 0.5) * 0.5  # Adjust with emotion regulation level

    return torch.tensor([state_anxiety, trait_anxiety])