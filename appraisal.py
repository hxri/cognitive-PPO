import torch
import torch.nn.functional as F
import numpy as np
import torch

def motivational_relevance(obs):
    """Computes motivational relevance for a batch of observations.
    Motivational relevance is a function of the L1 distance to the goal.
    Some observation states do not contain the goal, so relevance is zero.
    """

    # print(obs.shape)
    batch_size, w, _ = obs.size()
    relevance = torch.zeros(batch_size)
    agent_pos = torch.nonzero(obs == 10)[:, 1:]
    goal_poss = torch.nonzero(obs == 8)
    for goal in goal_poss:
        idx, goal_pos = goal[0], goal[1:]
        dist = torch.norm(agent_pos[idx] - goal_pos.float(), 1)
        relevance[idx] = 1 - (dist - 1) / (2 * (w - 1))
    norm = 1/(1+torch.exp(relevance))
    return norm

def novelty(logits):
    """Computes novelty according to the KL Divergence from perfect uncertainty.
    The higher the KL Divergence, the less novel the scenario,
    so we take novelty as the negative of the KL Divergence.
    """
    batch_size, num_actions = logits.size()
    P = torch.softmax(logits, dim=1)
    Q = torch.full(P.size(), 1 / num_actions)
    nov = -torch.sum(Q * torch.log(Q / P), dim=1)
    norm = 1/(1+torch.exp(nov))
    return norm

def certainity(logits): # Entropy
    """
    Computes the coping potential of an RL agent given the logits predicted by the model.

    Args:
        logits: A tensor of shape (batch_size, num_actions) containing the logits for each action.

    Returns:
        A tensor of shape (batch_size,) containing the coping potential for each example in the batch.
    """
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    norm = 1/(1+torch.exp(entropy))
    return norm

def coping_potential(logits):
    """
    The coping potential measures the degree of control that the agent has over
    the environment, based on the difference between the expected and actual
    outcomes of its actions. A higher coping potential means that the agent has
    more control over the environment, while a lower coping potential means
    that the agent has less control.
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)

    # Calculate expected reward
    expected_reward = torch.sum(probs * logits, dim=1)

    # Calculate actual reward
    actual_reward = torch.max(logits, dim=1).values

    # Calculate coping potential
    coping_pot = expected_reward - actual_reward
    norm = 1/(1+torch.exp(coping_pot))
    return norm

def anticipation(logits):
    """
    The function takes in a tensor logits which represents the predicted logits
    for each action in a given state. We convert these logits to probabilities
    using the softmax function. Then, we calculate the entropy of the
    probabilities, which measures the amount of uncertainty or randomness in
    the probability distribution. The anticipation is defined as the inverse
    of the entropy, which gives a measure of the agent's level of confidence or
    expectation about the outcomes of the actions. Finally, we return the
    tensor of anticipations.
    """

    # Convert logits to probabilities using softmax function
    probs = torch.softmax(logits, dim=1)
    
    # Calculate the entropy of the probabilities
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    
    # Calculate the anticipation as the inverse of the entropy
    anticipation = 1.0 / entropy
    # Return the tensor of anticipations
    norm = 1/(1+torch.exp(anticipation))
    return norm


def goal_congruence(logits):
    """
    To compute the goal congruence score for an RL agent in a grid world environment, goal congruence of an RL agent in
    a grid world environment using entropy, we can compute the entropy of the policy distribution over actions at each
    state in the grid world. The idea is that if the agent's policy is more focused on the actions that lead to the
    goal state, the entropy of the policy distribution should be lower.

    The function first converts the observation and logits to PyTorch tensors. It then computes the softmax of the
    logits to get the policy distribution over actions, and computes the entropy of the policy distribution using the
    formula -sum(p*log(p)), where p is the probability of each action. The function also computes the KL-divergence
    from the uniform distribution as a measure of how far the policy is from being uniformly distributed.

    Finally, the function computes the goal congruence score as the negative entropy of the policy distribution,
    adjusted by the KL-divergence from uniform distribution, and returns it as a NumPy array.

    Args:
        obs (numpy array): Observation from the environment, representing the current state.
        logits (numpy array): Predicted logits for the action probabilities at the current state.

    Returns:
        goal_congruence (float): Goal congruence score, computed as the entropy of the policy distribution over actions.
    """

    # # Convert obs and logits to PyTorch tensors
    # obs = torch.Tensor(obs).unsqueeze(0)
    # logits = torch.Tensor(logits).unsqueeze(0)

    # Compute the softmax of the logits to get the policy distribution over actions
    policy = torch.softmax(logits, dim=1)

    # Compute the entropy of the policy distribution
    entropy = -torch.sum(policy * torch.log(policy), dim=1)

    # Compute the distance from uniform distribution
    n_actions = policy.shape[1]
    uniform_policy = torch.ones_like(policy) / n_actions
    kl_divergence = torch.sum(policy * torch.log(policy / uniform_policy), dim=1)

    # Compute the goal congruence score as the negative entropy of the policy distribution
    goal_congruence = -(entropy + kl_divergence)
    # print(goal_congruence.shape)
    # Return the goal congruence score as a PyTorch tensor
    norm = 1/(1+torch.exp(goal_congruence))
    return norm