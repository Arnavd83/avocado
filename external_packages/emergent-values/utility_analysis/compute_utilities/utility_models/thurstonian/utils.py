from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import numpy as np

# ===================== THURSTONIAN HELPER FUNCTIONS ===================== #

def fit_thurstonian_model(graph: 'PreferenceGraph', num_epochs: int = 1000, learning_rate: float = 0.01) -> Tuple[Dict[Any, Dict[str, float]], float, float]:
    """
    Fits the Thurstonian model using the preference graph.
    
    Args:
        graph: PreferenceGraph object containing the preference data
        num_epochs: Number of epochs for optimization
        learning_rate: Learning rate for optimization
        
    Returns:
        Tuple containing:
        - option_utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
        - model_log_loss: The log loss (cross-entropy) of the fitted model
        - model_accuracy: Training accuracy of the model
    """
    option_id_to_idx = {option['id']: idx for idx, option in enumerate(graph.options)}
    n_options = len(graph.options)
    
    # Initialize parameters
    mu = torch.randn(n_options, requires_grad=True) * 0.01
    s = torch.randn(n_options, requires_grad=True) * 0.01
    mu = torch.nn.Parameter(mu.clone())
    s = torch.nn.Parameter(s.clone())
    
    optimizer = torch.optim.Adam([mu, s], lr=learning_rate)
    
    # Prepare training data
    idx_A_list = []
    idx_B_list = []
    probs_A_list = []
    
    # Convert graph edges to training data
    for edge in graph.edges.values():
        A_id = edge.option_A['id']
        B_id = edge.option_B['id']
        idx_A = option_id_to_idx[A_id]
        idx_B = option_id_to_idx[B_id]
        
        idx_A_list.append(idx_A)
        idx_B_list.append(idx_B)
        probs_A_list.append(edge.probability_A)
    
    # Convert to tensors
    idx_A_tensor = torch.tensor(idx_A_list, dtype=torch.long)
    idx_B_tensor = torch.tensor(idx_B_list, dtype=torch.long)
    labels_tensor = torch.tensor(probs_A_list, dtype=torch.float32)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Normalize mu to have mean zero and variance one
        mu_mean = torch.mean(mu)
        mu_std = torch.std(mu) + 1e-5
        mu_normalized = (mu - mu_mean) / mu_std
        
        # Adjust sigma^2 accordingly
        scaling_factor = 1 / (mu_std + 1e-5)
        sigma2 = torch.exp(s)
        sigma2_normalized = sigma2 * (scaling_factor ** 2)
        
        # Extract normalized parameters for the pairs
        mu_A = mu_normalized[idx_A_tensor]
        mu_B = mu_normalized[idx_B_tensor]
        sigma2_A = sigma2_normalized[idx_A_tensor]
        sigma2_B = sigma2_normalized[idx_B_tensor]
        
        variance = sigma2_A + sigma2_B + 1e-5
        delta = mu_A - mu_B
        
        z = delta / torch.sqrt(variance)
        
        # Compute probabilities using the CDF of the normal distribution
        normal = torch.distributions.Normal(0, 1)
        prob_A = normal.cdf(z)
        
        # Compute loss
        loss = F.binary_cross_entropy(prob_A, labels_tensor, reduction='mean')
        
        if torch.isnan(loss):
            print("Loss is NaN at epoch:", epoch)
            break
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
        loss.backward()
        optimizer.step()
        
    # Get final parameters
    with torch.no_grad():
        mu_mean = torch.mean(mu)
        mu_std = torch.std(mu) + 1e-5
        mu_normalized = (mu - mu_mean) / mu_std
        scaling_factor = 1 / (mu_std + 1e-5)
        sigma2 = torch.exp(s)
        sigma2_normalized = sigma2 * (scaling_factor ** 2)
        
    mu_np = mu_normalized.detach().numpy()
    sigma2_np = sigma2_normalized.detach().numpy()
    
    # Create utilities dictionary
    option_utilities = {
        option['id']: {'mean': mu_i, 'variance': sigma2_i}
        for option, mu_i, sigma2_i in zip(graph.options, mu_np, sigma2_np)
    }
    
    # Compute metrics
    y_true = labels_tensor.numpy()
    mu_A = mu_np[idx_A_list]
    mu_B = mu_np[idx_B_list]
    sigma2_A = sigma2_np[idx_A_list]
    sigma2_B = sigma2_np[idx_B_list]
    variance = sigma2_A + sigma2_B + 1e-5
    delta = mu_A - mu_B
    z = delta / np.sqrt(variance)
    prob_A = normal.cdf(torch.tensor(z)).numpy().astype(np.float64)
    
    # Compute log loss
    eps = 1e-5
    prob_A = np.clip(prob_A, eps, 1 - eps)
    model_log_loss = -np.mean(y_true * np.log(prob_A) + (1 - y_true) * np.log(1 - prob_A))
    
    # Convert both predicted and ground truth probabilities to binary labels using 0.5 threshold
    y_pred_binary = (prob_A >= 0.5).astype(float)
    y_true_binary = (y_true >= 0.5).astype(float)
    model_accuracy = np.mean(y_pred_binary == y_true_binary)
    
    return option_utilities, model_log_loss, model_accuracy

def evaluate_thurstonian_model(
    graph: 'PreferenceGraph',
    utilities: Dict[Any, Dict[str, float]],
    edge_indices: List[Tuple[Any, Any]]
) -> Dict[str, float]:
    """
    Evaluate Thurstonian model's goodness-of-fit on the given edges.
    
    Args:
        graph: PreferenceGraph object containing the preference data
        utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
        edge_indices: List of (option_A_id, option_B_id) tuples to evaluate on
        
    Returns:
        Dictionary containing evaluation metrics:
        - log_loss: Cross-entropy loss between predicted and actual probabilities
        - accuracy: Accuracy of binary predictions (prob >= 0.5)
    """
    # Extract actual probabilities and prepare data for predictions
    y_true = []
    mu_A_list = []
    mu_B_list = []
    sigma2_A_list = []
    sigma2_B_list = []
    
    for A_id, B_id in edge_indices:
        edge_index = (A_id, B_id)  # Maintain original orientation
        if edge_index not in graph.edges:
            continue
            
        edge = graph.edges[edge_index]
        
        # Get actual probability (ensuring consistent ordering)
        if edge.option_A['id'] == A_id:
            prob_A = edge.probability_A
        else:
            prob_A = 1 - edge.probability_A
            
        y_true.append(prob_A)
        
        # Get utilities
        mu_A_list.append(utilities[A_id]['mean'])
        mu_B_list.append(utilities[B_id]['mean'])
        sigma2_A_list.append(utilities[A_id]['variance'])
        sigma2_B_list.append(utilities[B_id]['variance'])
    
    if not y_true:  # No valid edges found
        return {'log_loss': float('nan'), 'accuracy': float('nan')}
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    mu_A = np.array(mu_A_list)
    mu_B = np.array(mu_B_list)
    sigma2_A = np.array(sigma2_A_list)
    sigma2_B = np.array(sigma2_B_list)
    
    # Compute predicted probabilities
    variance = sigma2_A + sigma2_B + 1e-5
    delta = mu_A - mu_B
    z = delta / np.sqrt(variance)
    normal = torch.distributions.Normal(0, 1)
    prob_A = normal.cdf(torch.tensor(z)).numpy()
    
    # Compute metrics
    eps = 1e-5
    prob_A = np.clip(prob_A, eps, 1 - eps)
    model_log_loss = -np.mean(y_true * np.log(prob_A) + (1 - y_true) * np.log(1 - prob_A))
    
    # Convert both predicted and ground truth probabilities to binary labels using 0.5 threshold
    y_pred_binary = (prob_A >= 0.5).astype(float)
    y_true_binary = (y_true >= 0.5).astype(float)
    model_accuracy = np.mean(y_pred_binary == y_true_binary)
    
    return {
        'log_loss': float(model_log_loss),
        'accuracy': float(model_accuracy)
    }
