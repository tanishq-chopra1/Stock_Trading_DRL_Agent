"""
Proximal Policy Optimization (PPO) Agent
Implementation from scratch using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO
    Shared feature extraction with separate heads for policy and value
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        hidden_dims: List[int] = [256, 128],
        activation: str = 'tanh',
        dropout: float = 0.0
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('tanh', 'relu')
            dropout: Dropout probability for regularization (0.0 = no dropout)
        """
        super(ActorCritic, self).__init__()
        
        # Choose activation
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout_prob = dropout
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
            ])
            # Add dropout after each hidden layer (except last)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Linear(prev_dim, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Smaller initialization for policy head
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            action_probs: Action probabilities [batch_size, action_dim]
            value: State value [batch_size, 1]
        """
        features = self.shared_net(state)
        
        # Actor: action logits -> probabilities
        action_logits = self.actor(features)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # Critic: state value
        value = self.critic(features)
        
        return action_probs, value
    
    def get_action_and_value(
        self, 
        state: torch.Tensor,
        action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log prob, entropy, and value
        Used during training
        
        Args:
            state: State tensor
            action: Action tensor (if provided, compute log prob for this action)
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: State value
        """
        action_probs, value = self.forward(state)
        
        dist = Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.0,
        device: str = 'cpu'
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping threshold
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability for regularization
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Create actor-critic network
        self.ac_network = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=lr)
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(
        self, 
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action given state
        
        Args:
            state: Environment state
            deterministic: If True, select argmax action. If False, sample from distribution
            
        Returns:
            action: Selected action
            value: Value estimate
            log_prob: Log probability of action
        """
        # Set to eval mode if deterministic (disables dropout)
        was_training = self.ac_network.training
        if deterministic:
            self.ac_network.eval()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.ac_network(state_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
                # Restore training mode if it was on
                if was_training:
                    self.ac_network.train()
                # For deterministic actions, we don't store trajectory data
                return action.item(), value.item(), 0.0
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        # Store trajectory (only when not deterministic)
        self.states.append(state)
        self.actions.append(action.item())
        self.values.append(value.item())
        self.log_probs.append(log_prob.item())
        
        return action.item(), value.item(), log_prob.item()
    
    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for state after last state
            
        Returns:
            advantages: Advantage estimates
            returns: Discounted returns
        """
        advantages = []
        gae = 0
        
        # Work backwards from last timestep
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values, dtype=np.float32)
        
        return advantages, returns
    
    def update(
        self,
        n_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Update policy using PPO
        
        Args:
            n_epochs: Number of epochs to update
            batch_size: Batch size for updates
            
        Returns:
            Dictionary of training metrics
        """
        # Compute advantages
        advantages, returns = self.compute_gae(
            self.rewards,
            self.values,
            self.dones,
            next_value=0.0
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Multiple epochs of SGD
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(n_epochs):
            # Create random indices for mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, values = self.ac_network.get_action_and_value(
                    batch_states,
                    batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = ((values.squeeze() - batch_returns) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.value_coef * value_loss + 
                    self.entropy_coef * entropy_loss
                )
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Clear trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'total_loss': (total_policy_loss + self.value_coef * total_value_loss + self.entropy_coef * total_entropy) / n_updates
        }
    
    def save(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'ac_network': self.ac_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ac_network.load_state_dict(checkpoint['ac_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
