'''Neural CFR bot for Toss or Hold'em (IAP 2026).'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
import torch.multiprocessing as mp
from collections import deque
import os
from typing import List, Tuple, Dict, Any, Optional
import pkrbot
from dataclasses import dataclass
import time
import itertools

NUM_ACTIONS = 9
MAX_HOLE_CARDS = 3

_CHEN_RANKS = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
               "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
_CHEN_BASE = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0, 10: 5.0, 9: 4.5, 8: 4.0,
              7: 3.5, 6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}

def _chen_score_local(c1, c2) -> float:
    def _rc(card):
        t = str(card).strip()
        if t.startswith("10"):
            return "T"
        return t[0].upper() if t and t[0].upper() in _CHEN_RANKS else "2"

    def _sc(card):
        t = str(card).strip()
        return t[-1].lower() if t else ""

    r1, r2 = _CHEN_RANKS.get(_rc(c1), 2), _CHEN_RANKS.get(_rc(c2), 2)
    hi, lo = max(r1, r2), min(r1, r2)
    pair = (r1 == r2)
    s1, s2 = _sc(c1), _sc(c2)
    suited = bool(s1) and bool(s2) and s1 == s2
    gap = hi - lo
    score = float(_CHEN_BASE.get(hi, 1.0))
    if pair:
        score *= 2.0
        if score < 5.0:
            score = 5.0
    else:
        if suited:
            score += 2.0
        if gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        elif gap == 4:
            score -= 5.0
        elif gap >= 5:
            score -= 6.0
        if gap <= 1 and hi <= 11:
            score += 1.0
        if lo <= 5 and hi <= 11 and not suited:
            score -= 0.5
    return max(0.0, min(20.0, score))

def _ncfr_bool_env(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or '').strip().lower()
    if not raw:
        return default
    return raw not in ('0', 'false', 'no', 'off')

def _ncfr_model_dir() -> str:
    return (os.environ.get('NCFR_MODEL_DIR') or '').strip() or "complete_neural_cfr_models"

def _ncfr_torch_save(obj, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

def _ncfr_float_env(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name, '').strip()
        return float(raw) if raw else float(default)
    except Exception:
        return float(default)

def _ncfr_hand_shape(cards):
    try:
        cards = list(cards or [])
        if not cards:
            return (0.0, 0.0, 0.0)
        ranks = sorted((_RANK_TO_INT_NCFR.get(str(c)[0].upper(), 2) for c in cards), reverse=True)
        suits = [str(c)[1].lower() for c in cards if len(str(c)) >= 2]

        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        mx_suit = max(suit_counts.values()) if suit_counts else 0
        suited_class = 2 if mx_suit >= 3 else (1 if mx_suit == 2 else 0)

        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        mx_rank = max(rank_counts.values()) if rank_counts else 1
        pair_class = 2 if mx_rank >= 3 else (1 if mx_rank == 2 else 0)

        gap = (max(ranks) - min(ranks)) if ranks else 0
        gap_bucket = 0
        for i, e in enumerate((4, 7, 10)):
            if gap <= e:
                gap_bucket = i
                break
        else:
            gap_bucket = 3
        return (pair_class / 2.0, suited_class / 2.0, gap_bucket / 3.0)
    except Exception:
        return (0.0, 0.0, 0.0)

_RANK_TO_INT_NCFR = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
                     "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

def _ncfr_in_position(street: int, active: int) -> bool:
    if street == 0:
        return active == 1
    return active == 0
MAX_BOARD_CARDS = 6
MAX_TOTAL_CARDS = MAX_HOLE_CARDS + MAX_BOARD_CARDS

@dataclass
class PrioritizedExperience:
    priority: float
    experience: Dict
    index: int
    
    def __lt__(self, other):
        return self.priority < other.priority

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
       
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by num_heads"
       
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)
       
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
       
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        residual = x
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            m = mask
            if m.dim() == 2:
                m = m.unsqueeze(1).unsqueeze(2)
            elif m.dim() == 3:
                m = m.unsqueeze(1)
            m = (m != 0)
            m = m | (~m.any(dim=-1, keepdim=True))
            neg = torch.finfo(attention_scores.dtype).min / 2
            attention_scores = attention_scores.masked_fill(~m, neg)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = torch.nan_to_num(attention_probs, nan=0.0)
        attention_probs = self.dropout(attention_probs)
        
        attention_output = torch.matmul(attention_probs, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attention_output = self.o_proj(attention_output)
        attention_output = self.dropout(attention_output)
        
        x = self.layer_norm1(residual + attention_output)
        
        residual = x
        x = self.ffn(x)
        x = self.layer_norm2(residual + x)
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class HierarchicalStateEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.card_embedding = nn.Embedding(52, embed_dim)
        self.position_embedding = nn.Embedding(MAX_TOTAL_CARDS, embed_dim)
        
        self.card_transformer = TransformerEncoder(embed_dim, num_heads, num_layers, dropout)
        
        self.hand_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.game_encoder = nn.Sequential(
            nn.Linear(50, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim)
        )
        
        self.final_projection = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, card_indices, positions, game_features, attention_mask=None):
        target_dtype = next(self.game_encoder.parameters()).dtype
        if game_features.dtype != target_dtype:
            game_features = game_features.to(dtype=target_dtype)
        
        batch_size = card_indices.shape[0]
        
        card_embeds = self.card_embedding(card_indices)
        pos_embeds = self.position_embedding(positions)
        card_embeds = card_embeds + pos_embeds
        
        card_features = self.card_transformer(card_embeds, attention_mask)
        
        hand_features = card_features[:, :MAX_HOLE_CARDS, :]
        board_features = card_features[:, MAX_HOLE_CARDS:, :]
        
        _kpm = None
        if attention_mask is not None:
            _kpm = (attention_mask[:, MAX_HOLE_CARDS:] == 0)
            _kpm = _kpm & ~_kpm.all(dim=-1, keepdim=True)
        hand_context, _ = self.hand_attention(
            hand_features,
            board_features,
            board_features,
            key_padding_mask=_kpm,
        )

        if attention_mask is not None:
            hand_mask = attention_mask[:, :MAX_HOLE_CARDS].unsqueeze(-1).to(dtype=hand_features.dtype)
            board_mask = attention_mask[:, MAX_HOLE_CARDS:].unsqueeze(-1).to(dtype=board_features.dtype)
            masked_hand = hand_context * hand_mask
            masked_board = board_features * board_mask
            hand_count = hand_mask.sum(dim=1).clamp(min=1e-8)
            board_count = board_mask.sum(dim=1).clamp(min=1e-8)
            hand_pooled = masked_hand.sum(dim=1) / hand_count
            board_pooled = masked_board.sum(dim=1) / board_count
        else:
            hand_pooled = torch.mean(hand_context, dim=1)
            board_pooled = torch.mean(board_features, dim=1)
        
        game_encoded = self.game_encoder(game_features)
        
        combined = torch.cat([hand_pooled, board_pooled, game_encoded], dim=1)
        state_representation = self.final_projection(combined)
        state_representation = torch.nan_to_num(state_representation, nan=0.0)
        return state_representation

class DistributionalQNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, num_atoms=51, hidden_dim=512):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_atoms)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_actions * num_atoms)
        )
        
        self.register_buffer('tau', torch.linspace(0, 1, num_atoms + 1)[1:].view(1, -1))
        
    def forward(self, state):
        features = self.feature_extractor(state)
        
        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.num_actions, self.num_atoms)
        
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_atoms = torch.clamp(q_atoms, -800.0, 800.0)
        
        return q_atoms
    
    def get_q_values(self, state):
        q_atoms = self.forward(state)
        return q_atoms.mean(dim=-1)

class StochasticPolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=512):
        super().__init__()
        self.num_actions = num_actions
        
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        self.baseline_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

        with torch.no_grad():
            final = self.policy_network[-1]
            prior = torch.full((num_actions,), 0.05)
            if num_actions >= 6:
                prior[0] = 0.12
                prior[1] = 0.40
                prior[2:min(6, num_actions)] = 0.08
            prior = prior / prior.sum()
            final.bias.copy_(torch.log(prior))

    def forward(self, state, legal_mask=None, training=True):
        logits = self.policy_network(state)
        
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, -1e9)
        
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        
        temperature = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)
        scaled_logits = logits / (temperature + 1e-8)
        scaled_logits = torch.nan_to_num(scaled_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        
        if training:
            uniform = torch.rand_like(scaled_logits).clamp_(1e-20, 1-1e-20)
            gumbel_noise = -torch.log(-torch.log(uniform))
            noisy_logits = scaled_logits + gumbel_noise
            noisy_logits = torch.nan_to_num(noisy_logits, nan=0.0)
            action_probs = F.softmax(noisy_logits, dim=-1)
        else:
            action_probs = F.softmax(scaled_logits, dim=-1)
        
        action_probs = torch.nan_to_num(action_probs, nan=1e-12)
        action_probs = action_probs.clamp_min(1e-12)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        
        log_probs = torch.log(action_probs + 1e-12)
        
        baseline = self.baseline_network(state)
        baseline = torch.nan_to_num(baseline, nan=0.0, posinf=100.0, neginf=-100.0)
        
        return {
            'logits': logits,
            'action_probs': action_probs,
            'log_probs': log_probs,
            'temperature': temperature,
            'baseline': baseline
        }

class NeuralCounterfactualRegretMinimizer(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=256):
        super().__init__()
        
        self.regret_predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        self.cfv_predictor = nn.Sequential(
            nn.Linear(state_dim + num_actions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, state, current_strategy=None):
        regret_pred = self.regret_predictor(state)
        
        advantages = self.advantage_network(state)
        
        cfv_pred = None
        if current_strategy is not None:
            strategy_input = torch.cat([state, current_strategy], dim=1)
            cfv_pred = self.cfv_predictor(strategy_input)
        
        return {
            'regret_pred': regret_pred,
            'advantages': advantages,
            'cfv_pred': cfv_pred
        }
    
    def compute_regret_matching_strategy(self, regrets, legal_mask=None):
        positive_regrets = F.relu(regrets)
        
        if legal_mask is not None:
            positive_regrets = positive_regrets.masked_fill(legal_mask == 0, 0.0)
        
        sum_positive = positive_regrets.sum(dim=-1, keepdim=True)
        
        uniform_strategy = torch.ones_like(regrets)
        if legal_mask is not None:
            uniform_strategy = uniform_strategy.masked_fill(legal_mask == 0, 0.0)
            uniform_strategy = uniform_strategy / uniform_strategy.sum(dim=-1, keepdim=True)
        else:
            uniform_strategy = uniform_strategy / uniform_strategy.shape[-1]
        
        strategy = torch.where(
            sum_positive > 0,
            positive_regrets / (sum_positive + 1e-8),
            uniform_strategy
        )
        
        return strategy

class AverageStrategyNetwork(nn.Module):

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, state: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.net(state)
        logits = torch.clamp(logits, -10.0, 10.0)
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, -1e9)
        return F.softmax(logits, dim=-1)

    class PrioritizedReservoirBuffer:
        def __init__(self, capacity: int, alpha=0.6):
            self.capacity = capacity
            self.alpha = alpha
            self.data = []
            self.priorities = []
            self.n_seen = 0
    
        def add(self, item, priority=1.0):
            self.n_seen += 1
            if len(self.data) < self.capacity:
                self.data.append(item)
                self.priorities.append(priority)
                return
            j = random.randint(0, self.n_seen - 1)
            if j < self.capacity:
                self.data[j] = item
                self.priorities[j] = priority
    
        def sample(self, batch_size: int):
            if not self.data:
                return []
            priorities = np.array(self.priorities)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.data), min(batch_size, len(self.data)), p=probs, replace=False)
            return [self.data[i] for i in indices]
    
        def __len__(self):
            return len(self.data)
            
class ReservoirBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data = []
        self.n_seen = 0
    def add(self, item: Dict[str, Any]):
        self.n_seen += 1
        if len(self.data) < self.capacity:
            self.data.append(item)
            return
        j = random.randint(0, self.n_seen - 1)
        if j < self.capacity:
            self.data[j] = item
    def sample(self, batch_size: int):
        if not self.data:
            return []
        k = min(int(batch_size), len(self.data))
        return random.sample(self.data, k)
    def __len__(self):
        return len(self.data)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.max_priority = 1.0
        
    def push(self, experience, priority=None):
        if priority is None:
            priority = self.max_priority
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None, None
            
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        samples = [self.buffer[i] for i in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority.item() if torch.is_tensor(priority) else priority
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self):
        return len(self.buffer)

class NStepTDLearning:
    def __init__(self, n_steps=5, gamma=0.99, lambda_=0.7):
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_ = lambda_
        self.trajectory_buffer = []
        
    def compute_gae(self, rewards, values, next_values, dones):
        batch_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = next_values[t] if not dones[t] else 0
            else:
                next_value = values[t + 1] if not dones[t] else 0
                
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * self.lambda_ * (1 - dones[t]) * last_gae
            last_gae = advantages[t]
            
        returns = advantages + values
        return advantages, returns
    
class StrategyCombiner(nn.Module):
    def __init__(self, state_dim, num_sources=3, hidden_dim=128):
        super().__init__()
        self.num_sources = num_sources
        self.combiner_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_sources)
        )
        
    def forward(self, state_rep):
        raw_weights = self.combiner_network(state_rep)
        weights = F.softmax(raw_weights, dim=-1)
        return weights

class HandStrengthNetwork(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.card_embedding = nn.Embedding(52, embed_dim)
        self.position_embedding = nn.Embedding(MAX_TOTAL_CARDS, embed_dim)
        
        self.transformer = TransformerEncoder(embed_dim, num_heads=8, num_layers=3)
        
        self.strength_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, card_indices, positions):
        card_embeds = self.card_embedding(card_indices)
        pos_embeds = self.position_embedding(positions)
        embeds = card_embeds + pos_embeds
        
        features = self.transformer(embeds)
        pooled = torch.mean(features, dim=1)
        strength = self.strength_predictor(pooled)
        return strength

class CompleteNeuralCFRBot(Bot):
    
    def __init__(self):
        super().__init__()

        self.verbose = os.environ.get('BOT_VERBOSE', '0').strip() not in {'', '0', 'false', 'False', 'FALSE'}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.verbose:
            print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            if self.verbose:
                print(f"GPU: {torch.cuda.get_device_name()}")
            torch.backends.cudnn.benchmark = True
            
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(_ncfr_model_dir(), "tb_logs")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            if self.verbose:
                print(f"[logs] TensorBoard logs: {log_dir} (run 'tensorboard --logdir {log_dir}' to view)")
        except Exception as e:
            print(f"[logs] Failed to initialize TensorBoard writer: {e}")
            self.writer = None

        _prec = (os.environ.get('NCFR_PRECISION') or 'auto').strip().lower()
        _cc_major = 0
        if self.device.type == 'cuda':
            try:
                _cc_major = int(torch.cuda.get_device_properties(0).major)
            except Exception:
                _cc_major = 0
        _bf16_native = _cc_major >= 8
        if _prec == 'bf16':
            self.use_bf16 = self.device.type == 'cuda'
        elif _prec == 'fp32':
            self.use_bf16 = False
        else:
            self.use_bf16 = (self.device.type == 'cuda'
                             and torch.cuda.is_bf16_supported())
        if self.use_bf16:
            print(f"[mixed precision] Using BF16 (sm_{_cc_major}x, native={_bf16_native})")
        else:
            print("[mixed precision] FP32")
        
        self.num_actions = int(NUM_ACTIONS)

        self.card_to_index = self._create_card_mapping()
        
        self.state_encoder = HierarchicalStateEncoder().to(self.device)
        self.q_network = DistributionalQNetwork(128, self.num_actions).to(self.device)
        self.target_q_network = DistributionalQNetwork(128, self.num_actions).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.strategy_combiner = StrategyCombiner(128, num_sources=3).to(self.device)

        self.combiner_street_prior = nn.Embedding(7, 3).to(self.device)
        self.hand_strength_network = HandStrengthNetwork().to(self.device)
        self.hand_strength_optimizer = optim.AdamW(self.hand_strength_network.parameters(), lr=1e-4, weight_decay=1e-5)
        try:
            _hs = os.path.join(_ncfr_model_dir(), 'hand_strength.pt')
            if os.path.exists(_hs):
                _blob = torch.load(_hs, map_location=self.device, weights_only=False)
                self.hand_strength_network.load_state_dict(_blob['state_dict'])
                self.hand_strength_network.eval()
                self._hand_strength_trained = True
                print(f"[hand-strength] loaded distilled weights "
                      f"(val_mse={_blob.get('val_mse', float('nan')):.5f} vs "
                      f"baseline {_blob.get('baseline_mse', float('nan')):.5f})", flush=True)
            else:
                self._hand_strength_trained = False
        except Exception as _e:
            self._hand_strength_trained = False
            print(f"[hand-strength] could not load distilled weights: {_e}", flush=True)
        self.policy_network = StochasticPolicyNetwork(128, self.num_actions).to(self.device)
        self.cfr_networks = nn.ModuleList([
            NeuralCounterfactualRegretMinimizer(128, self.num_actions).to(self.device),
            NeuralCounterfactualRegretMinimizer(128, self.num_actions).to(self.device),
        ])
        self.avg_strategy_network = AverageStrategyNetwork(128, self.num_actions).to(self.device)

        self.raise_sizing_head = nn.Sequential(
            nn.Linear(128 + 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        ).to(self.device)
        
        if self.use_bf16:
            models_to_bf16 = [
                self.state_encoder, self.q_network, self.target_q_network,
                self.policy_network, self.strategy_combiner, self.combiner_street_prior,
                self.hand_strength_network, *self.cfr_networks,
                self.avg_strategy_network, self.raise_sizing_head
            ]
            for m in models_to_bf16:
                m.to(dtype=torch.bfloat16)
        
        self.encoder_optimizer = optim.AdamW(self.state_encoder.parameters(), lr=1e-4, weight_decay=1e-5)
        self.q_optimizer = optim.AdamW(self.q_network.parameters(), lr=1e-4, weight_decay=1e-5)
        _critic_ids = {id(q) for q in self.policy_network.baseline_network.parameters()}
        self._actor_params = (
            [q for q in self.policy_network.parameters() if id(q) not in _critic_ids]
            + list(self.raise_sizing_head.parameters()))
        self.policy_optimizer = optim.AdamW(
            self._actor_params,
            lr=1e-4,
            weight_decay=1e-5
        )
        self.value_optimizer = optim.AdamW(
            self.policy_network.baseline_network.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
        )
        self.cfr_optimizers = [
            optim.AdamW(self.cfr_networks[0].parameters(), lr=1e-4, weight_decay=1e-5),
            optim.AdamW(self.cfr_networks[1].parameters(), lr=1e-4, weight_decay=1e-5),
        ]
        self.avg_strategy_optimizer = optim.AdamW(self.avg_strategy_network.parameters(), lr=1e-4, weight_decay=1e-5)
        self.combiner_optimizer = optim.AdamW(
            list(self.strategy_combiner.parameters()) + list(self.combiner_street_prior.parameters()),
            lr=1e-4,
            weight_decay=1e-5,
        )
        
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        self.td_learner = NStepTDLearning(n_steps=5, gamma=0.99, lambda_=0.7)
        self.gamma = 0.99
        self.strategy_cache = {}
        
        self.current_hand_experiences = []
        self.opponent_action_history = []
        self.board_history = []
        self.strategy_weights_history = deque(maxlen=1000)
        self.hand_context = {
            'hand_id': 0,
            'street': 0,
            'actions_taken': 0,
            'my_actions': [],
            'opponent_actions': []
        }
        
        self.batch_size = int(os.environ.get('NCFR_BATCH_SIZE', '512') or 512)
        self.accum_steps = 2
        self.accum_count = 0
        self._accum_counts = {}
        self.target_update_freq = 1000
        self.learning_steps = 0
        self.training_mode = os.environ.get('NCFR_TRAINING', '1').strip().lower() not in ('0', 'false', 'no')
        self._last_training_mode = bool(self.training_mode)

        self.train_combiner = True
        self.combiner_entropy_coef = 0.05
        self.combiner_prior_kl_coef = 0.01

        self.value_driven_discard = True
        self.discard_eval_temperature = 0.5
        self.discard_train_temperature = 1.0

        self.print_training = True
        self.print_every_seconds = 30.0
        self.print_every_steps = 200
        self._last_progress_print_time = 0.0
        self._last_progress_print_step = 0
        self._last_buffer_wait_print = 0.0
        self._last_losses: Dict[str, float] = {}

        self.use_deep_mccfr = True
        self.mccfr_traversals_per_hand = 1
        self.mccfr_train_every_hands = 5

        self.stage2_steps = int(os.environ.get('NCFR_STAGE2_STEPS', '5000') or 0)
        self.stage3_steps = int(os.environ.get('NCFR_STAGE3_STEPS', '15000') or 0)
        self.mccfr_advantage_buffer = ReservoirBuffer(capacity=200000)
        self.mccfr_strategy_buffer = ReservoirBuffer(capacity=200000)
        self.mccfr_value_buffer = ReservoirBuffer(capacity=200000)
        self.mccfr_batch_size = 1024
        self.mccfr_value_batch_size = 512
        self.mccfr_chance_samples_per_hand = 2
        self.mccfr_discard_focus_samples = 1
        self.mccfr_avg_mix_anneal_updates = 5000
        self.mccfr_updates_seen = 0
        self.mccfr_weight_clip = 10.0
        self.mccfr_regret_clip_quantile = 0.95

        self.snapshot_every_hands = int(os.environ.get('NCFR_SNAPSHOT_EVERY_HANDS', '0') or 0)
        
        if self.training_mode and os.environ.get('NCFR_PREFILL', '').strip().lower() in ('1', 'true', 'yes'):
            self._pre_fill_buffers(num_sim_hands=100)

        self._optimizers_stepped = set()
        try:
            self.lr_schedulers = {
                'encoder': torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=5000, gamma=0.9),
                'q': torch.optim.lr_scheduler.StepLR(self.q_optimizer, step_size=5000, gamma=0.9),
                'policy': torch.optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=5000, gamma=0.9),
                'value': torch.optim.lr_scheduler.StepLR(self.value_optimizer, step_size=5000, gamma=0.9),
                'cfr0': torch.optim.lr_scheduler.StepLR(self.cfr_optimizers[0], step_size=5000, gamma=0.9),
                'cfr1': torch.optim.lr_scheduler.StepLR(self.cfr_optimizers[1], step_size=5000, gamma=0.9),
                'avg': torch.optim.lr_scheduler.StepLR(self.avg_strategy_optimizer, step_size=5000, gamma=0.9),
                'comb': torch.optim.lr_scheduler.StepLR(self.combiner_optimizer, step_size=5000, gamma=0.9),
            }
        except Exception:
            self.lr_schedulers = {}

        self.train_action_selection = 'sample'
        self.eval_action_selection = 'argmax'
        self.eval_temperature = 1.0
        self.min_action_prob = 1e-12
        
        _explore_default = 0.15 if self.training_mode else 0.0
        self.epsilon = _ncfr_float_env('NCFR_EPSILON', _explore_default)
        self.epsilon_decay = _ncfr_float_env('NCFR_EPSILON_DECAY', 0.99995)
        self.min_epsilon = _ncfr_float_env(
            'NCFR_EPSILON_MIN',
            min(0.02, self.epsilon) if self.training_mode else 0.0)
        
        self.load_models()

        if self.training_mode and os.environ.get('NCFR_EPSILON_MIN') is None:
            self.min_epsilon = min(self.min_epsilon, float(self.epsilon))

        self._sync_model_modes()

        try:
            if getattr(self, 'print_training', True):
                try:
                    _p = next(self.state_encoder.parameters())
                    _real = f"{str(_p.dtype).replace('torch.', '')} on {_p.device}"
                except Exception:
                    _real = "unknown"
                print(f"[bot] initialized | training_mode={bool(self.training_mode)} "
                      f"| device={self.device} | bf16={self.use_bf16} "
                      f"| ACTUAL params: {_real}", flush=True)
        except Exception:
            pass
        
        if self.verbose:
            print("Complete Neural CFR Bot initialized")

    def _sync_model_modes(self):
        modules = [
            self.state_encoder, self.q_network, self.target_q_network,
            self.strategy_combiner, self.hand_strength_network, self.policy_network,
            *self.cfr_networks, self.avg_strategy_network,
            self.raise_sizing_head
        ]
        mode_fn = nn.Module.train if self.training_mode else nn.Module.eval
        for m in modules:
            mode_fn(m)

    def _select_action_index(self, probs: torch.Tensor, mode: str, temperature: float = 1.0) -> int:
        probs = probs.float().clamp_min(0)
        if probs.sum().item() <= 0:
            return int(torch.argmax(probs).item())

        if mode == 'argmax':
            return int(torch.argmax(probs).item())

        t = max(float(temperature), 1e-8)
        if abs(t - 1.0) > 1e-6:
            scaled = torch.pow(probs.clamp_min(self.min_action_prob), 1.0 / t)
            probs = scaled / scaled.sum()
        else:
            probs = probs / probs.sum()

        return int(torch.multinomial(probs, 1).item())
    
    def _create_card_mapping(self):
        mapping = {}
        idx = 0
        for suit in ['c', 'd', 'h', 's']:
            for rank in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']:
                mapping[rank + suit] = idx
                idx += 1
        return mapping
    
    def encode_cards(self, cards):
        indices = []
        for card in cards:
            card_str = None
            if isinstance(card, str):
                card_str = card
            else:
                try:
                    rank_names = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
                    suit_names = ['c', 'd', 'h', 's']
                    card_str = rank_names[int(card.rank)] + suit_names[int(card.suit)]
                except Exception:
                    try:
                        card_str = str(card)
                    except Exception:
                        card_str = None

            if not card_str:
                indices.append(0)
                continue

            card_str = str(card_str).strip()
            if len(card_str) >= 2:
                r = card_str[0].upper()
                s = card_str[1].lower()
                card_str = r + s

            indices.append(self.card_to_index.get(card_str, 0))
        return indices
    
    def encode_state_tensor(self, round_state, active):
        my_cards = list(round_state.hands[active]) if getattr(round_state, 'hands', None) else []
        board_cards = list(getattr(round_state, 'board', []) or [])
        
        card_indices = []
        positions = []
        
        hand_indices = self.encode_cards(my_cards[:MAX_HOLE_CARDS])
        card_indices.extend(hand_indices)
        positions.extend(list(range(len(hand_indices))))
        while len(card_indices) < MAX_HOLE_CARDS:
            card_indices.append(0)
            positions.append(0)
    
        board_indices = self.encode_cards(board_cards[:MAX_BOARD_CARDS])
        card_indices.extend(board_indices)
        positions.extend([MAX_HOLE_CARDS + i for i in range(len(board_indices))])
    
        while len(card_indices) < MAX_TOTAL_CARDS:
            card_indices.append(0)
            positions.append(0)
        
        game_features = self._extract_game_features(round_state, active)
        
        card_tensor = torch.tensor([card_indices], dtype=torch.long, device=self.device)
        pos_tensor = torch.tensor([positions], dtype=torch.long, device=self.device)
        game_tensor = torch.from_numpy(game_features).unsqueeze(0).to(self.device)
        
        if self.use_bf16:
            game_tensor = game_tensor.to(dtype=torch.bfloat16)
        
        attention_mask = torch.zeros(1, MAX_TOTAL_CARDS, device=self.device, dtype=torch.bool)
        attention_mask[0, :min(MAX_HOLE_CARDS, len(hand_indices))] = 1
        attention_mask[0, MAX_HOLE_CARDS:MAX_HOLE_CARDS + len(board_indices)] = 1
        
        return {
            'card_indices': card_tensor,
            'positions': pos_tensor,
            'game_features': game_tensor,
            'attention_mask': attention_mask
        }
    
    def _extract_game_features(self, round_state, active):
        street = round_state.street
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        continue_cost = opp_pip - my_pip
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack
        current_pot = my_contribution + opp_contribution
        dealer_index = int(self.hand_context.get('dealer', 0))
        is_dealer = (active == dealer_index)
        
        min_raise, max_raise = 0, 0
        if hasattr(round_state, 'raise_bounds'):
            try:
                min_raise, max_raise = round_state.raise_bounds()
            except:
                pass
        
        board_cards = list(getattr(round_state, 'board', []) or [])
        in_discard_phase = 1.0 if int(street) in (2, 3) else 0.0
        hole_count = float(len(getattr(round_state, 'hands', [[], []])[active]) if getattr(round_state, 'hands', None) else 0)
        board_count = float(len(board_cards))

        features = [
            street / 6.0,
            my_stack / STARTING_STACK,
            opp_stack / STARTING_STACK,
            current_pot / STARTING_STACK,
            continue_cost / STARTING_STACK,
            
            1.0 if is_dealer else 0.0,
            my_pip / STARTING_STACK,
            opp_pip / STARTING_STACK,

            in_discard_phase,
            hole_count / float(MAX_HOLE_CARDS),
            board_count / float(MAX_BOARD_CARDS),
            
            my_contribution / (my_contribution + opp_contribution + 1e-8),
            opp_contribution / (my_contribution + opp_contribution + 1e-8),
            
            continue_cost / (current_pot + continue_cost + 1e-8) if continue_cost > 0 else 0.0,
            (my_stack - continue_cost) / (current_pot + 1e-8) if continue_cost > 0 else my_stack / (current_pot + 1e-8),
            
            min_raise / STARTING_STACK,
            max_raise / STARTING_STACK,
            (max_raise - min_raise) / STARTING_STACK if max_raise > min_raise else 0.0,
            
            my_stack / (current_pot + 1e-8),
            opp_stack / (current_pot + 1e-8),
            
            min(my_stack, opp_stack) / STARTING_STACK,
            max(my_stack, opp_stack) / STARTING_STACK,
            
            self.hand_context['actions_taken'] / 20.0,
            len(self.current_hand_experiences) / 10.0,
            
            self._calculate_aggression_metric(),
            
            self._estimate_hand_strength(round_state.hands[active], board_cards),

            1.0 if _ncfr_in_position(street, active) else 0.0,
            1.0 if (continue_cost > 0 and my_pip > 0) else 0.0,
            min(3.0, float(self.hand_context['opponent_actions'].count('raise'))) / 3.0,
            self._last_aggressor_feature(),
            *_ncfr_hand_shape(round_state.hands[active] if getattr(round_state, 'hands', None) else []),
            
            math.sin(2 * math.pi * self.hand_context['hand_id'] / 100.0),
            math.cos(2 * math.pi * self.hand_context['hand_id'] / 100.0),
            
            self._get_opponent_features(),
            
            self._get_action_sequence_features(),

            *self._board_texture_numeric(board_cards),
        ]
        
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50], dtype=np.float32)
        
    def _pre_fill_buffers(self, num_sim_hands=100):
        print(f"[init] Pre-filling buffers with {num_sim_hands} simulated hands...")
       
        self.accum_count = 0
        accum_steps = self.accum_steps
       
        for hand_num in range(num_sim_hands):
            mock_deltas = [random.randint(-STARTING_STACK, STARTING_STACK),
                           random.randint(-STARTING_STACK, STARTING_STACK)]
           
            mock_hands = [['As', 'Ks', 'Qs'], ['Ad', 'Kd', 'Qd']]
            mock_board = ['2h', '3h', '4h', '5h', '6h', '7h']
           
            mock_previous = RoundState(
                random.randint(0, 1),
                random.randint(0, 6),
                [random.randint(0, BIG_BLIND), random.randint(0, BIG_BLIND)],
                [STARTING_STACK - random.randint(0, BIG_BLIND), STARTING_STACK - random.randint(0, BIG_BLIND)],
                mock_hands,
                mock_board,
                None
            )
           
            mock_terminal = TerminalState(deltas=mock_deltas, previous_state=mock_previous)
           
            self._deep_mccfr_update_from_observed_hand(mock_terminal)
           
            mock_state_data = {
                'card_indices': torch.randint(0, 52, (1, MAX_TOTAL_CARDS)),
                'positions': torch.arange(MAX_TOTAL_CARDS).unsqueeze(0),
                'game_features': torch.rand(1, 50),
                'attention_mask': torch.ones(1, MAX_TOTAL_CARDS)
            }
            if self.use_bf16:
                mock_state_data['game_features'] = mock_state_data['game_features'].to(dtype=torch.bfloat16)
            mock_state_rep = torch.rand(128)
            if self.use_bf16:
                mock_state_rep = mock_state_rep.to(dtype=torch.bfloat16)
            mock_legal_mask = torch.ones(NUM_ACTIONS)
            mock_action_idx = random.randint(0, NUM_ACTIONS - 1)
            
            mock_exp = {
                'state': mock_state_rep.clone().float(),
                'action': mock_action_idx,
                'reward': float(mock_deltas[0]),
                'next_state': None,
                'next_legal_mask': None,
                'done': True,
                'return': float(mock_deltas[0]),
                'advantage': 0.0,
                'legal_mask': mock_legal_mask,
                'old_log_prob': random.uniform(-10, 0),
                'raise_fraction': random.uniform(0, 1) if mock_action_idx in range(2, 6) else None,
                
                'player': 0,
                'street': random.randint(0, 6),
                'state_data': mock_state_data,
                'action_type': random.choice(['fold', 'call', 'check', 'raise', 'discard']),
                'baseline_value': random.uniform(-1, 1),
                'timestamp': time.time(),
                'hand_id': hand_num,
            }
            
            mock_td_error = abs(random.uniform(-1, 1))
            self.replay_buffer.push(mock_exp, priority=mock_td_error)
           
            if self.accum_count == 0:
                self.policy_optimizer.zero_grad()
           
            mock_input = torch.rand(1, 128).to(self.device)
            mock_target = torch.rand(1, 1).to(self.device)
            if self.use_bf16:
                mock_input = mock_input.to(dtype=torch.bfloat16)
                mock_target = mock_target.to(dtype=torch.bfloat16)
           
            mock_out = self.policy_network(mock_input)
            mock_loss = F.mse_loss(mock_out['baseline'], mock_target) / accum_steps
           
            mock_loss.backward()
           
            self.accum_count += 1
            if self.accum_count % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self._actor_params, 0.5)
                self.policy_optimizer.step()
                self.policy_optimizer.zero_grad(set_to_none=True)
                self.accum_count = 0
           
            if hand_num % 5 == 0:
                print(f"[pre-fill] {hand_num}/{num_sim_hands} hands | Replay: {len(self.replay_buffer)} | MCCFR adv: {len(self.mccfr_advantage_buffer)}")
       
        self.value_optimizer.zero_grad(set_to_none=True)

        if len(self.replay_buffer) > self.replay_buffer.capacity:
            drop = int(len(self.replay_buffer) * 0.1)
            self.replay_buffer.buffer = self.replay_buffer.buffer[drop:]
            self.replay_buffer.priorities = self.replay_buffer.priorities[drop:]
            self.replay_buffer.position = len(self.replay_buffer.buffer) % self.replay_buffer.capacity
       
        print(f"[init] Pre-fill complete! Replay: {len(self.replay_buffer)} | MCCFR adv: {len(self.mccfr_advantage_buffer)}")

    def _calculate_aggression_metric(self):
        if not self.hand_context['my_actions']:
            return 0.5
        
        aggressive_actions = sum(1 for a in self.hand_context['my_actions'] if a in ['raise', 'bet'])
        total_actions = len(self.hand_context['my_actions'])
        
        return aggressive_actions / max(total_actions, 1)
    
    def _get_opponent_features(self):
        if not self.opponent_action_history:
            return 0.5
        
        opp_actions = [a for _, a in self.opponent_action_history if a is not None]
        if not opp_actions:
            return 0.5
        
        aggressive_count = sum(1 for a in opp_actions if a in ['raise', 'bet', 'reraise'])
        return aggressive_count / len(opp_actions)
    
    def _last_aggressor_feature(self) -> float:
        try:
            mine = self.hand_context.get('my_actions') or []
            theirs = self.hand_context.get('opponent_actions') or []
            my_last = max((i for i, a in enumerate(mine) if a == 'raise'), default=-1)
            opp_last = max((i for i, a in enumerate(theirs) if a == 'raise'), default=-1)
            if my_last < 0 and opp_last < 0:
                return 0.0
            return 0.5 if my_last >= opp_last else 1.0
        except Exception:
            return 0.0

    def _get_action_sequence_features(self):
        if not self.hand_context['my_actions']:
            return 0.5
        
        actions = self.hand_context['my_actions']
        if len(actions) < 2:
            return 0.5
        
        changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
        return 1.0 - (changes / (len(actions) - 1))
    
    def _board_texture_numeric(self, board_cards):
        if not board_cards:
            return [0.0] * 10

        ranks = "23456789TJQKA"
        rs = [ranks.index(str(c)[0]) for c in board_cards]
        ss = [str(c)[1] for c in board_cards]
        rs_sorted = sorted(rs)

        unique_ranks = len(set(rs))
        pairedness = 1.0 - (unique_ranks / len(rs))
        max_suit = max(ss.count(s) for s in set(ss))
        suitedness = max_suit / len(ss)

        gaps = 0
        for i in range(1, len(rs_sorted)):
            gaps += max(0, rs_sorted[i] - rs_sorted[i - 1] - 1)
        connectivity = 1.0 / (1.0 + gaps)

        high_card = max(rs) / 12.0
        low_card = min(rs) / 12.0
        span = max(rs_sorted) - min(rs_sorted)
        straight_potential = max(0.0, 1.0 - span / 12.0)

        return [
            len(board_cards) / float(MAX_BOARD_CARDS),
            unique_ranks / len(rs),
            pairedness,
            suitedness,
            connectivity,
            high_card,
            low_card,
            straight_potential,
            float(max_suit == len(ss)),
            float(pairedness > 0.0),
        ]

    def _round_state_key(self, rs: Any) -> Tuple:
        try:
            street = int(getattr(rs, 'street', 0))
        except Exception:
            street = 0
        try:
            button = int(getattr(rs, 'button', 0))
        except Exception:
            button = 0
        try:
            pips = tuple(int(x) for x in (getattr(rs, 'pips', [0, 0]) or [0, 0]))
        except Exception:
            pips = (0, 0)
        try:
            stacks = tuple(int(x) for x in (getattr(rs, 'stacks', [0, 0]) or [0, 0]))
        except Exception:
            stacks = (0, 0)
        try:
            board_n = len(getattr(rs, 'board', []) or [])
        except Exception:
            board_n = 0
        try:
            hands = getattr(rs, 'hands', None)
            h0 = len(hands[0]) if hands is not None and len(hands) > 0 else 0
            h1 = len(hands[1]) if hands is not None and len(hands) > 1 else 0
        except Exception:
            h0, h1 = 0, 0
        return (street, button, pips, stacks, board_n, h0, h1)

    def _infer_action_from_transition(self, prev_state: Any, curr_state: Any) -> Tuple[int, Optional[str]]:
        try:
            actor = int(getattr(prev_state, 'button', 0)) % 2
        except Exception:
            actor = 0

        try:
            prev_board = list(getattr(prev_state, 'board', []) or [])
            curr_board = list(getattr(curr_state, 'board', []) or [])
            prev_hands = getattr(prev_state, 'hands', None)
            curr_hands = getattr(curr_state, 'hands', None)
            if prev_hands is not None and curr_hands is not None:
                if len(curr_board) == len(prev_board) + 1 and len(curr_hands[actor]) == len(prev_hands[actor]) - 1:
                    return actor, 'discard'
        except Exception:
            pass

        pp = None
        cp = None
        try:
            pp = list(getattr(prev_state, 'pips', [0, 0]) or [0, 0])
            cp = list(getattr(curr_state, 'pips', [0, 0]) or [0, 0])
            if int(cp[actor]) > int(pp[actor]):
                opp = 1 - actor
                if int(pp[opp]) > int(pp[actor]) and int(cp[actor]) == int(pp[opp]):
                    return actor, 'call'
                return actor, 'raise'
        except Exception:
            pass

        try:
            if int(getattr(curr_state, 'street', 0)) != int(getattr(prev_state, 'street', 0)):
                if pp is not None:
                    opp = 1 - actor
                    if int(pp[opp]) - int(pp[actor]) > 0:
                        return actor, 'call'
                return actor, 'check'
        except Exception:
            pass

        return actor, None

    def _update_histories_from_round_state(self, round_state: Any) -> None:
        try:
            key = self._round_state_key(round_state)
        except Exception:
            return
        if getattr(self, '_last_seen_state_key', None) == key:
            return

        prev = getattr(round_state, 'previous_state', None)
        if prev is not None:
            try:
                actor, atype = self._infer_action_from_transition(prev, round_state)
                if atype is not None:
                    street = int(getattr(prev, 'street', getattr(round_state, 'street', 0)))
                    me = int(getattr(self, 'my_index', 0))
                    if actor == me:
                        self.hand_context['my_actions'].append(atype)
                    else:
                        self.opponent_action_history.append((street, atype))
                        self.hand_context['opponent_actions'].append(atype)
            except Exception:
                pass

        self._last_seen_state_key = key

    def _compute_combiner_weights(self, state_rep: torch.Tensor, street: int) -> torch.Tensor:
        w = self.strategy_combiner(state_rep)
        s = int(max(0, min(6, int(street))))
        prior_logits = self.combiner_street_prior(torch.tensor([s], device=state_rep.device))
        prior = F.softmax(prior_logits, dim=-1)
        w = w * prior
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return w

    def _value_driven_discard_choice(self, round_state: Any, active: int) -> Optional[Tuple[int, float]]:
        if not getattr(self, 'value_driven_discard', False):
            return None
        try:
            legal = round_state.legal_actions()
            if DiscardAction not in legal:
                return None
        except Exception:
            return None
        try:
            hand = list(getattr(round_state, 'hands', [[], []])[active])
            n = int(max(0, min(3, len(hand))))
        except Exception:
            return None
        if n <= 0:
            return None
    
        scores: List[float] = []
        valid: List[int] = []
        for i in range(int(n)):
            try:
                rs = copy.deepcopy(round_state)
                rs = rs.proceed(DiscardAction(int(i)))
                try:
                    if CheckAction in rs.legal_actions():
                        rs = rs.proceed(CheckAction())
                except Exception:
                    pass
    
                sd = self.encode_state_tensor(rs, active)
                with torch.inference_mode():
                    rep = self.state_encoder(
                        sd['card_indices'], sd['positions'], sd['game_features'], sd['attention_mask']
                    )
                    out = self.policy_network(rep, legal_mask=None, training=False)
                    v_raw = out['baseline'].view(-1)[0]
    
                    v = float(v_raw.item())
                    if not math.isfinite(v):
                        v = 0.0
                scores.append(v)
                valid.append(int(i))
            except Exception as e:
                continue
    
        if not valid:
            return None
    
        t = float(self.discard_train_temperature if self.training_mode else self.discard_eval_temperature)
        t = max(t, 1e-8)
    
        s = torch.tensor(scores, dtype=torch.float32, device=self.device)
    
        logits = s / t
    
        if logits.numel() > 1:
            logits = logits - torch.max(logits)
    
        probs = F.softmax(logits, dim=-1)
    
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp_min(1e-12)
        probs = probs / probs.sum().clamp_min(1e-12)
    
        if self.training_mode and self.train_action_selection == 'sample':
            j = int(torch.multinomial(probs, 1).item())
        else:
            j = int(torch.argmax(probs).item())
    
        chosen = valid[j]
        logp = float(torch.log(probs[j]).item())
    
        return chosen, logp
    
    def _blend_ready(self) -> bool:
        s3 = int(getattr(self, 'stage3_steps', 0) or 0)
        if s3 <= 0:
            return True
        return int(getattr(self, 'learning_steps', 0)) >= s3

    def get_action(self, game_state, round_state, active):
        if bool(self.training_mode) != bool(getattr(self, '_last_training_mode', False)):
            self._sync_model_modes()
            self._last_training_mode = bool(self.training_mode)
    
        try:
            self.hand_context['street'] = int(getattr(round_state, 'street', 0))
        except Exception:
            self.hand_context['street'] = 0
    
        if not hasattr(self, 'my_index'):
            try:
                self.my_index = int(active)
            except Exception:
                self.my_index = 0
    
        self._update_histories_from_round_state(round_state)
    
        legal_actions = round_state.legal_actions()
        
        state_data = self.encode_state_tensor(round_state, active)
        
        if self.use_bf16:
            state_data['game_features'] = state_data['game_features'].to(dtype=torch.bfloat16)
        
        with torch.inference_mode():
            state_rep = self.state_encoder(
                state_data['card_indices'],
                state_data['positions'],
                state_data['game_features'],
                state_data['attention_mask']
            )
            if self.use_bf16:
                state_rep = state_rep.to(dtype=torch.bfloat16)
        
        legal_mask = self._create_complete_legal_mask(legal_actions, round_state, active)
        
        with torch.inference_mode():
            policy_output = self.policy_network(state_rep, legal_mask, training=False)
            policy_strategy = policy_output['action_probs']
            policy_strategy = policy_strategy * legal_mask
            ps_sum = policy_strategy.sum(dim=-1, keepdim=True)
            if (ps_sum <= 0).any():
                policy_strategy = legal_mask / legal_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            else:
                policy_strategy = policy_strategy / ps_sum
    
            q_values = self.q_network.get_q_values(state_rep)
            q_logits = q_values / (policy_output['temperature'] + 1e-8)
            q_logits = q_logits.masked_fill(legal_mask == 0, -1e9)
            q_strategy = F.softmax(q_logits, dim=-1)
            q_strategy = q_strategy * legal_mask
            q_sum = q_strategy.sum(dim=-1, keepdim=True)
            if (q_sum <= 0).any():
                q_strategy = legal_mask / legal_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            else:
                q_strategy = q_strategy / q_sum
    
            cfr_strategy = self.avg_strategy_network(state_rep, legal_mask)
        
        with torch.inference_mode():
            combiner_weights = self._compute_combiner_weights(state_rep, int(getattr(round_state, 'street', 0)))
        
        combined_strategy = (
            combiner_weights[0, 0] * q_strategy +
            combiner_weights[0, 1] * policy_strategy +
            combiner_weights[0, 2] * cfr_strategy
        )
    
        combined_strategy = combined_strategy * legal_mask
        cs_sum = combined_strategy.sum(dim=-1, keepdim=True)
        if (cs_sum <= 0).any():
            combined_strategy = legal_mask / legal_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        else:
            combined_strategy = combined_strategy / cs_sum
        
        self.strategy_weights_history.append({
            'q_weight': combiner_weights[0, 0].item(),
            'policy_weight': combiner_weights[0, 1].item(),
            'cfr_weight': combiner_weights[0, 2].item()
        })
        
        raise_fraction = None
        behavior_log_prob = None
        behavior_value = float(policy_output['baseline'][0].item())
    
        if self.training_mode:
            selection_mode = self.train_action_selection
            _forced = (os.environ.get('NCFR_TRAIN_FROM_BLEND') or '').strip()
            if _forced:
                _use_blend = _ncfr_bool_env('NCFR_TRAIN_FROM_BLEND', True)
            else:
                _use_blend = self._blend_ready()
            base_probs = combined_strategy[0] if _use_blend else policy_strategy[0]
        else:
            selection_mode = self.eval_action_selection
            base_probs = combined_strategy[0] if self._blend_ready() else policy_strategy[0]
    
        base_probs = base_probs * legal_mask[0]
        if float(base_probs.sum().item()) > 0:
            base_probs = base_probs / base_probs.sum()
        else:
            base_probs = legal_mask[0] / legal_mask[0].sum().clamp_min(1.0)
    
        probs_work = base_probs.detach().clone()
        action_idx = None
        behavior_log_prob = None
        raise_fraction = None
    
        selection_temperature = 1.0 if self.training_mode else float(self.eval_temperature)
    
        if DiscardAction in legal_actions:
            choice = self._value_driven_discard_choice(round_state, active)
            if choice is not None:
                di, dlogp = choice
                action_idx = 6 + int(di)
                behavior_log_prob = float(dlogp)
                raise_fraction = None
                action = DiscardAction(int(di))
    
        for _ in range(int(self.num_actions)):
            if action_idx is not None:
                break
            if probs_work.sum().item() <= 0:
                break
    
            if self.training_mode and self.epsilon > 0.0 and random.random() < self.epsilon:
                _explore = torch.nonzero(probs_work > 0, as_tuple=False).view(-1)
                if int(_explore.numel()) > 0:
                    _pick = int(torch.randint(int(_explore.numel()), (1,)).item())
                    chosen_idx = int(_explore[_pick].item())
                else:
                    chosen_idx = self._select_action_index(
                        probs_work, selection_mode, temperature=selection_temperature)
            else:
                chosen_idx = self._select_action_index(probs_work, selection_mode, temperature=selection_temperature)
    
            normed = probs_work / probs_work.sum()
            behaviour = normed
            if self.training_mode and self.epsilon > 0.0:
                _legal = (probs_work > 0).float()
                _n_legal = float(_legal.sum().item())
                if _n_legal > 0:
                    behaviour = ((1.0 - self.epsilon) * normed
                                 + self.epsilon * (_legal / _n_legal))
            chosen_logp = float(torch.log(
                behaviour[chosen_idx].clamp_min(self.min_action_prob)).item())
    
            candidate_raise_fraction = None
            candidate_logp = chosen_logp
    
            if 2 <= chosen_idx <= 5 and RaiseAction in legal_actions:
                bucket = int(chosen_idx - 2)
                if self.training_mode:
                    candidate_raise_fraction, frac_logp = self._sample_raise_fraction(state_rep, bucket)
                    candidate_logp += float(frac_logp)
                else:
                    candidate_raise_fraction = self._mean_raise_fraction(state_rep, bucket)
    
            candidate_action = self._convert_to_engine_action(
                chosen_idx, legal_actions, round_state, state_rep, candidate_raise_fraction
            )
    
            if candidate_action is not None:
                action_idx = chosen_idx
                behavior_log_prob = candidate_logp
                raise_fraction = candidate_raise_fraction
                action = candidate_action
                break
    
            probs_work[chosen_idx] = 0.0
    
        if action_idx is None:
            if CheckAction in legal_actions:
                action = CheckAction()
                action_idx = 1
            elif CallAction in legal_actions:
                action = CallAction()
                action_idx = 1
            elif FoldAction in legal_actions:
                action = FoldAction()
                action_idx = 0
            else:
                action = CheckAction()
                action_idx = 1
            behavior_log_prob = 0.0
        
        action_type = self._classify_action(action)
        self.hand_context['my_actions'].append(action_type)
        self.hand_context['actions_taken'] += 1
        
        if self.training_mode:
            self._store_td_experience(
                state_data, state_rep, legal_mask, action_idx, action_type,
                old_log_prob=behavior_log_prob,
                baseline_value=behavior_value,
                raise_fraction=raise_fraction,
                round_state=round_state,
            )
        
        return action
    
    def _create_complete_legal_mask(self, legal_actions, round_state, active):
        mask = torch.zeros(self.num_actions, device=self.device)
        
        if FoldAction in legal_actions:
            mask[0] = 1

        if CallAction in legal_actions or CheckAction in legal_actions:
            mask[1] = 1

        if RaiseAction in legal_actions:
            try:
                min_raise, max_raise = round_state.raise_bounds()
                if max_raise >= min_raise and max_raise > 0:
                    mask[2] = mask[3] = mask[4] = mask[5] = 1
            except Exception:
                pass

        if DiscardAction in legal_actions:
            try:
                n = len(round_state.hands[active]) if getattr(round_state, 'hands', None) else 0
                n = int(max(0, min(MAX_HOLE_CARDS, n)))
                for i in range(n):
                    mask[6 + i] = 1
            except Exception:
                pass
        
        return mask.unsqueeze(0)
    
    def _convert_to_engine_action(self, action_idx, legal_actions, round_state, state_rep, raise_fraction=None):
        try:
            acting_player = int(getattr(round_state, 'button', 0)) % 2
        except Exception:
            acting_player = 0

        if action_idx == 0:
            return FoldAction() if FoldAction in legal_actions else None
        
        elif action_idx == 1:
            if CallAction in legal_actions:
                return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()
            return None
        
        elif 2 <= action_idx <= 5:
            if RaiseAction not in legal_actions:
                return None
            
            try:
                min_raise, max_raise = round_state.raise_bounds()
            except Exception:
                return None

            if max_raise < min_raise or max_raise <= 0:
                return None
            
            if raise_fraction is None:
                bucket = int(max(0, min(3, action_idx - 2)))
                raise_fraction = self._mean_raise_fraction(state_rep, bucket)
            
            raise_amount = min_raise + int((max_raise - min_raise) * raise_fraction)
            
            raise_amount = max(min_raise, min(max_raise, raise_amount))
            
            return RaiseAction(raise_amount)

        elif 6 <= action_idx <= 8:
            if DiscardAction not in legal_actions:
                return None
            card_idx = int(action_idx - 6)
            try:
                if getattr(round_state, 'hands', None) is not None and card_idx < len(round_state.hands[acting_player]):
                    return DiscardAction(card_idx)
            except Exception:
                return None

        return None

    def _beta_params(self, state_rep: torch.Tensor, bucket: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bucket_onehot = torch.zeros(4, device=state_rep.device, dtype=state_rep.dtype)
        bucket_onehot[bucket] = 1.0
        bucket_onehot = bucket_onehot.unsqueeze(0)

        x = torch.cat([state_rep, bucket_onehot], dim=-1)
        raw = self.raise_sizing_head(x)

        raw = torch.nan_to_num(raw, nan=0.0, posinf=5.0, neginf=-5.0)

        one = torch.tensor(1.0, dtype=raw.dtype, device=raw.device)
        alpha = F.softplus(raw[:, 0]) + one
        beta = F.softplus(raw[:, 1]) + one

        return alpha.squeeze(0), beta.squeeze(0)

    def _sample_raise_fraction(self, state_rep: torch.Tensor, bucket: int) -> Tuple[float, float]:
        alpha, beta = self._beta_params(state_rep, bucket)

        min_val = torch.tensor(1.1, device=alpha.device, dtype=alpha.dtype)
        alpha = alpha.clamp_min(min_val)
        beta = beta.clamp_min(min_val)

        if not self.training_mode:
            fraction = alpha / (alpha + beta)
            fraction = fraction.clamp(0.0, 1.0)
            return float(fraction.item()), 0.0

        dist = Beta(alpha, beta)
        frac_tensor = dist.sample()
        frac_clamped = frac_tensor.clamp(min=1e-6, max=1-1e-6)
        log_prob = dist.log_prob(frac_clamped)
        fraction = frac_tensor.clamp(0.0, 1.0)

        return float(fraction.item()), float(log_prob.item())

    def _mean_raise_fraction(self, state_rep: torch.Tensor, bucket: int) -> float:
        alpha, beta = self._beta_params(state_rep, bucket)
        mean = alpha / (alpha + beta)
        return float(mean.item())

    def _estimate_hand_strength(self, hand_cards, board_cards):
        try:
            hole = [c for c in list(hand_cards or []) if c is not None]
            board = [c for c in list(board_cards or []) if c is not None]
            if len(board) >= 3 and len(hole) >= 2:
                import pkrbot as _pk
                def _mk(c):
                    return c if isinstance(c, _pk.Card) else _pk.Card(str(c))
                import itertools as _it
                _b = [_mk(c) for c in board]
                _h = [_mk(c) for c in hole]
                if len(_h) > 2:
                    score = max(int(_pk.evaluate(_b + list(_c)))
                                for _c in _it.combinations(_h, 2))
                else:
                    score = int(_pk.evaluate(_b + _h))
                return max(0.0, min(1.0, (score - 1_000_000) / 9_000_000.0))
            if len(hole) >= 2:
                best = 0.0
                for i in range(len(hole)):
                    for j in range(i + 1, len(hole)):
                        best = max(best, float(_chen_score_local(hole[i], hole[j])))
                return max(0.0, min(1.0, best / 20.0))
        except Exception:
            pass

        hand_indices = self.encode_cards(list(hand_cards)[:MAX_HOLE_CARDS])
        board_indices = self.encode_cards(list(board_cards)[:MAX_BOARD_CARDS])

        all_indices: List[int] = []
        positions: List[int] = []

        all_indices.extend(hand_indices)
        positions.extend(list(range(len(hand_indices))))
        while len(all_indices) < MAX_HOLE_CARDS:
            all_indices.append(0)
            positions.append(0)

        all_indices.extend(board_indices)
        positions.extend([MAX_HOLE_CARDS + i for i in range(len(board_indices))])
        while len(all_indices) < MAX_TOTAL_CARDS:
            all_indices.append(0)
            positions.append(0)
        
        with torch.inference_mode():
            card_tensor = torch.tensor([all_indices], dtype=torch.long, device=self.device)
            pos_tensor = torch.tensor([positions], dtype=torch.long, device=self.device)
            
            strength = self.hand_strength_network(card_tensor, pos_tensor).item()
        
        return max(0.0, min(1.0, strength))

    TREE_MAX_DEPTH = 3
    TREE_MAX_NODES = 400
    TREE_RAISE_FRACTIONS = (0.15, 0.35, 0.60, 1.00)

    def _create_legal_mask(self, legal_actions, round_state, active) -> torch.Tensor:
        return self._create_complete_legal_mask(legal_actions, round_state, active)

    def _index_to_action(self, action_idx, legal_actions, round_state):
        try:
            idx = int(action_idx)
        except Exception:
            return None
        frac = self.TREE_RAISE_FRACTIONS[idx - 2] if 2 <= idx <= 5 else None
        try:
            return self._convert_to_engine_action(idx, legal_actions, round_state,
                                                  None, raise_fraction=frac)
        except Exception:
            return None

    def _encode_state_rep(self, round_state, active) -> Optional[torch.Tensor]:
        try:
            sd = self.encode_state_tensor(round_state, active)
            with torch.no_grad():
                return self.state_encoder(sd['card_indices'], sd['positions'],
                                          sd['game_features'], sd['attention_mask'])
        except Exception:
            return None

    def _policy_strategy_for_state(self, round_state, active):
        rep = self._encode_state_rep(round_state, active)
        if rep is None:
            return np.ones(int(self.num_actions), dtype=np.float64) / float(self.num_actions)
        try:
            with torch.no_grad():
                out = self.policy_network(rep, training=False)
            return out['action_probs'].float().cpu().numpy()[0]
        except Exception:
            return np.ones(int(self.num_actions), dtype=np.float64) / float(self.num_actions)

    def _leaf_value(self, round_state, active) -> float:
        rep = self._encode_state_rep(round_state, active)
        if rep is None:
            return 0.0
        try:
            with torch.no_grad():
                out = self.policy_network(rep, training=False)
            return float(out['baseline'].float().reshape(-1)[0].item())
        except Exception:
            return 0.0

    def _calculate_action_value(self, round_state, active, action) -> float:
        return float(self._calculate_action_utility(round_state, active, action, 1.0, depth=0))

    def _reconstruct_round_state(self, state_data, active):
        try:
            snap = (state_data or {}).get('raw_state')
            if not snap:
                return None
            return RoundState(int(snap['button']), int(snap['street']),
                              list(snap['pips']), list(snap['stacks']),
                              [list(h) for h in snap['hands']],
                              list(snap['board']), None)
        except Exception:
            return None

    def _train_with_proper_targets(self, experiences) -> None:
        if not experiences:
            return
        reps, targets = [], []
        for exp in experiences:
            rep, tgt = exp.get('state_rep'), exp.get('td_target')
            if rep is None or tgt is None:
                continue
            reps.append(rep)
            targets.append(float(tgt))
        if not reps:
            return
        try:
            param = next(self.policy_network.parameters())
            states = torch.stack(reps).to(device=self.device, dtype=param.dtype)
            if states.dim() == 1:
                states = states.unsqueeze(0)
            y = torch.tensor(targets, device=self.device, dtype=torch.float32)
            out = self.policy_network(states, training=True)
            baseline = out['baseline'].float().reshape(-1)
            if baseline.shape[0] != y.shape[0]:
                return
            loss = F.mse_loss(baseline, y)
            self.value_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.baseline_network.parameters(), 1.0)
            self.value_optimizer.step()
            self._last_losses['td_value_loss'] = float(loss.item())
            self._optimizers_stepped.add('value')
        except Exception:
            return

    def _calculate_action_utility(self, round_state, active, action, reach_prob, depth=0):
        if depth == 0:
            self._tree_nodes_visited = 0
        try:
            new_state = round_state.proceed(action)
        except Exception:
            return 0.0

        if isinstance(new_state, TerminalState):
            try:
                return float(new_state.deltas[active])
            except Exception:
                return 0.0

        self._tree_nodes_visited = getattr(self, '_tree_nodes_visited', 0) + 1
        if depth + 1 >= self.TREE_MAX_DEPTH or self._tree_nodes_visited >= self.TREE_MAX_NODES:
            return self._leaf_value(new_state, active)

        strategy = self._policy_strategy_for_state(new_state, active)
        legal = new_state.legal_actions()
        legal_mask = self._create_legal_mask(legal, new_state, active)

        expected_utility = 0.0
        total_p = 0.0
        for action_idx in range(int(self.num_actions)):
            if legal_mask[0, action_idx] == 0:
                continue
            p = float(strategy[action_idx]) if action_idx < len(strategy) else 0.0
            if p <= 0.0:
                continue
            next_action = self._index_to_action(action_idx, legal, new_state)
            if next_action is None:
                continue
            child_utility = self._calculate_action_utility(
                new_state, active, next_action, reach_prob * p, depth + 1
            )
            expected_utility += p * child_utility
            total_p += p

        if total_p <= 0.0:
            return self._leaf_value(new_state, active)
        return float(expected_utility / total_p)
    
    def _classify_action(self, action):
        if isinstance(action, FoldAction):
            return 'fold'
        elif isinstance(action, CallAction):
            return 'call'
        elif isinstance(action, CheckAction):
            return 'check'
        elif isinstance(action, RaiseAction):
            return 'raise'
        elif isinstance(action, DiscardAction):
            return 'discard'
        return 'unknown'
    
    def _store_td_experience(self, state_data, state_rep, legal_mask, action_idx, action_type, old_log_prob: float, baseline_value: float, raise_fraction=None, round_state=None):
        raw_state = None
        if round_state is not None:
            try:
                raw_state = {
                    'button': int(round_state.button),
                    'street': int(round_state.street),
                    'pips': [int(x) for x in round_state.pips],
                    'stacks': [int(x) for x in round_state.stacks],
                    'hands': [[str(c) for c in h] for h in round_state.hands],
                    'board': [str(c) for c in (round_state.board or [])],
                }
            except Exception:
                raw_state = None
        experience = {
            'state_data': {**{k: v.cpu() if torch.is_tensor(v) else v
                              for k, v in state_data.items()},
                           'raw_state': raw_state},
            'state_rep': state_rep.detach().cpu().squeeze(0),
            'legal_mask': legal_mask.detach().cpu().squeeze(0),
            'action_idx': action_idx,
            'action_type': action_type,
            'old_log_prob': float(old_log_prob),
            'baseline_value': float(baseline_value),
            'raise_fraction': None if raise_fraction is None else float(raise_fraction),
            'timestamp': time.time(),
            'hand_id': self.hand_context['hand_id'],
            'street': self.hand_context['street']
        }
        
        self.current_hand_experiences.append(experience)
    
    def handle_new_round(self, game_state, round_state, active):
        if self.training_mode:
            print(f"[debug] Starting/Ending hand {game_state.round_num}")
        try:
            dealer_index = int(round_state.button) % 2
        except Exception:
            dealer_index = 0

        sb_index = dealer_index
        bb_index = 1 - dealer_index
        try:
            p0, p1 = int(round_state.pips[0]), int(round_state.pips[1])
            if p0 == SMALL_BLIND and p1 == BIG_BLIND:
                sb_index, bb_index = 0, 1
            elif p1 == SMALL_BLIND and p0 == BIG_BLIND:
                sb_index, bb_index = 1, 0
        except Exception:
            pass

        self.hand_context = {
            'hand_id': game_state.round_num,
            'street': round_state.street,
            'actions_taken': 0,
            'my_actions': [],
            'opponent_actions': [],
            'dealer': int(dealer_index),
            'sb_index': int(sb_index),
            'bb_index': int(bb_index),
        }
        
        self.current_hand_experiences = []
        self.opponent_action_history = []
        self.board_history = []
        self._last_seen_state_key = None
        
    def handle_round_over(self, game_state, terminal_state, active):

        if self.training_mode:
            self._deep_mccfr_update_from_observed_hand(terminal_state)
            self.mccfr_updates_seen = int(getattr(self, 'mccfr_updates_seen', 0)) + 1
            self.mccfr_hands_seen = int(getattr(self, 'mccfr_hands_seen', 0)) + 1
            every = max(1, int(getattr(self, 'mccfr_train_every_hands', 5)))
            if self.mccfr_hands_seen % every == 0:
                self._train_deep_mccfr_networks()

        if self.training_mode:
            self._process_td_experiences_and_push(terminal_state, active)
        
        if self.training_mode:
            self._train_all_networks(game_state)
        
        if self.training_mode:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        self.current_hand_experiences = []
        self.hand_context['my_actions'] = []
        self.hand_context['opponent_actions'] = []

    def _train_all_networks(self, game_state: Optional[GameState] = None):
        prev_learning_steps = int(getattr(self, 'learning_steps', 0))
        self._train_networks()

        if int(getattr(self, 'learning_steps', 0)) != prev_learning_steps:
            self._step_schedulers()

        if self.training_mode and getattr(self, 'print_training', True):
            now = time.time()
            step = int(getattr(self, 'learning_steps', 0))
            last_t = float(getattr(self, '_last_progress_print_time', 0.0))
            last_s = int(getattr(self, '_last_progress_print_step', 0))

            should_print = False
            if now - last_t >= float(getattr(self, 'print_every_seconds', 10.0)):
                should_print = True
            if step - last_s >= int(getattr(self, 'print_every_steps', 50)):
                should_print = True

            if should_print:
                def _lr(opt):
                    try:
                        return float(opt.param_groups[0].get('lr', 0.0))
                    except Exception:
                        return 0.0

                hand_id = None
                try:
                    hand_id = int(game_state.round_num) if game_state is not None else int(self.hand_context.get('hand_id', -1))
                except Exception:
                    hand_id = -1

                parts = [
                    f"[train] hand={hand_id} steps={step} buffer={len(self.replay_buffer)}/{int(getattr(self, 'batch_size', 0))}",
                    f"lr(enc)={_lr(self.encoder_optimizer):.2e}",
                    f"lr(q)={_lr(self.q_optimizer):.2e}",
                    f"lr(pi)={_lr(self.policy_optimizer):.2e}",
                ]
                if getattr(self, '_last_losses', None):
                    loss_items = []
                    for k in ['q_loss', 'policy_loss', 'value_loss', 'entropy', 'mccfr_adv_loss', 'mccfr_strat_loss', 'mccfr_value_loss']:
                        if k in self._last_losses:
                            loss_items.append(f"{k}={self._last_losses[k]:.4g}")
                    if loss_items:
                        parts.append("losses: " + ", ".join(loss_items))
                try:
                    hist = getattr(self, 'strategy_weights_history', None)
                    if hist:
                        recent = list(hist)[-200:]
                        n = float(len(recent))
                        wq = sum(float(h['q_weight']) for h in recent) / n
                        wp = sum(float(h['policy_weight']) for h in recent) / n
                        wc = sum(float(h['cfr_weight']) for h in recent) / n
                        parts.append(f"blend: q={wq:.2f} ppo={wp:.2f} cfr={wc:.2f}")
                except Exception:
                    pass
                print(" | ".join(parts), flush=True)

                self._last_progress_print_time = now
                self._last_progress_print_step = step
        if self.training_mode and game_state is not None:
            try:
                if int(getattr(self, 'snapshot_every_hands', 0)) > 0 and game_state.round_num % int(self.snapshot_every_hands) == 0:
                    self.save_models(snapshot_round=int(game_state.round_num))
            except Exception:
                pass
            
        save_every = int(os.environ.get('NCFR_SAVE_EVERY_STEPS', '500') or 500)
        last_saved = getattr(self, '_last_saved_step', None)
        if (self.training_mode and save_every > 0
                and self.learning_steps % save_every == 0
                and self.learning_steps != last_saved):
            self._last_saved_step = self.learning_steps
            print(f"[save] Auto-snapshot at step {self.learning_steps}")
            self.save_models(snapshot_round=self.learning_steps)

    def _step_schedulers(self):
        if not getattr(self, 'lr_schedulers', None):
            return
        stepped = getattr(self, '_optimizers_stepped', set())
        for key, sch in self.lr_schedulers.items():
            if key not in stepped:
                continue
            try:
                sch.step()
            except Exception:
                pass

    def _board_cards_from_state(self, round_state) -> List[str]:
        try:
            board = list(getattr(round_state, 'board', []) or [])
        except Exception:
            return []
        return [str(c) for c in board]

    def _process_td_experiences_and_push(self, terminal_state, active):
        if not self.current_hand_experiences:
            return
        exps = self.current_hand_experiences
        T = len(exps)
        rewards = torch.zeros(T, dtype=torch.float32)
        dones = torch.zeros(T, dtype=torch.float32)
        values = torch.tensor([float(e.get('baseline_value', 0.0)) for e in exps], dtype=torch.float32)
        rewards[-1] = float(terminal_state.deltas[active])
        dones[-1] = 1.0
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        next_values[-1] = 0.0
        advantages, returns = self.td_learner.compute_gae(rewards, values, next_values, dones)
        
        reward_clip = 400.0
        return_clip = 800.0
        advantage_clip = 8.0
        
        for i in range(T):
            state = exps[i]['state_rep'].clone().float()
            legal_mask = exps[i]['legal_mask'].clone().float()
            action = int(exps[i]['action_idx'])
            next_state = None
            next_legal_mask = None
            if i < T - 1:
                next_state = exps[i + 1]['state_rep'].clone().float()
                try:
                    next_legal_mask = exps[i + 1]['legal_mask'].clone().float()
                except Exception:
                    next_legal_mask = None
            
            clipped_reward = torch.clamp(rewards[i], -reward_clip, reward_clip)
            clipped_return = torch.clamp(returns[i], -return_clip, return_clip)
            clipped_advantage = torch.clamp(advantages[i], -advantage_clip, advantage_clip)
            
            sample = {
                'state': state,
                'state_data': exps[i].get('state_data'),
                'next_state_data': (exps[i + 1].get('state_data')
                                    if i < T - 1 else None),
                'action': action,
                'reward': float(clipped_reward.item()),
                'next_state': next_state,
                'next_legal_mask': next_legal_mask,
                'done': bool(dones[i].item() > 0.5),
                'return': float(clipped_return.item()),
                'advantage': float(clipped_advantage.item()),
                'legal_mask': legal_mask,
                'old_log_prob': float(exps[i].get('old_log_prob', 0.0)),
                'raise_fraction': exps[i].get('raise_fraction', None),
                'player': int(active),
                'street': int(exps[i].get('street', 0)),
            }
            self.replay_buffer.push(sample)

    def _calculate_recent_opponent_aggression(self) -> float:
        hist = getattr(self, 'opponent_action_history', None)
        if not hist:
            return 0.5

        try:
            opp_actions = [a for _, a in hist if a is not None]
        except Exception:
            opp_actions = []

        if not opp_actions:
            return 0.5

        max_n = int(getattr(self, 'opp_aggr_window', 24))
        half_life = float(getattr(self, 'opp_aggr_half_life', 8.0))

        seq = opp_actions[-max_n:]
        if not seq:
            return 0.5

        weights: List[float] = []
        scores: List[float] = []
        for i, a in enumerate(reversed(seq)):
            at = str(a).lower()
            if at == 'discard':
                continue

            if at in {'raise', 'bet', 'reraise'}:
                s = 1.0
            elif at == 'call':
                s = 0.35
            elif at == 'check':
                s = 0.0
            elif at == 'fold':
                s = 0.15
            else:
                s = 0.25

            w = math.exp(-float(i) / max(1e-6, half_life))
            weights.append(w)
            scores.append(s)

        if not weights:
            return 0.5

        num = sum(w * s for w, s in zip(weights, scores))
        den = sum(weights)
        return float(max(0.0, min(1.0, num / max(1e-9, den))))

    def _process_td_experiences(self, terminal_state, active):
        if not self.current_hand_experiences:
            return
        
        experiences_with_returns = []
        
        for i, exp in enumerate(reversed(self.current_hand_experiences)):
            round_state = self._reconstruct_round_state(exp['state_data'], active)
            if round_state is None:
                continue
            
            state_value = self._calculate_state_value(round_state, active)
            
            action_values = np.zeros(int(self.num_actions), dtype=np.float32)
            for action_idx in range(int(self.num_actions)):
                if exp['legal_mask'][0, action_idx] == 0:
                    continue
                    
                action = self._index_to_action(action_idx, round_state.legal_actions(), round_state)
                if action is None:
                    continue
                    
                action_values[action_idx] = self._calculate_action_value(
                    round_state, active, action
                )
            
            exp_with_return = exp.copy()
            exp_with_return['state_value'] = state_value
            exp_with_return['action_values'] = action_values
            
            if i == 0:
                exp_with_return['td_target'] = terminal_state.deltas[active]
            else:
                next_state_value = experiences_with_returns[-1]['state_value']
                exp_with_return['td_target'] = next_state_value
            
            experiences_with_returns.append(exp_with_return)
        
        self._train_with_proper_targets(experiences_with_returns)

    def _calculate_state_value(self, round_state, active, depth=0):
        if isinstance(round_state, TerminalState):
            try:
                return float(round_state.deltas[active])
            except Exception:
                return 0.0

        if depth == 0:
            self._tree_nodes_visited = 0
        if depth >= self.TREE_MAX_DEPTH or getattr(self, '_tree_nodes_visited', 0) >= self.TREE_MAX_NODES:
            return self._leaf_value(round_state, active)

        strategy = self._policy_strategy_for_state(round_state, active)
        legal = round_state.legal_actions()
        legal_mask = self._create_legal_mask(legal, round_state, active)

        expected_value = 0.0
        total_p = 0.0
        for action_idx in range(int(self.num_actions)):
            if legal_mask[0, action_idx] == 0:
                continue
            p = float(strategy[action_idx]) if action_idx < len(strategy) else 0.0
            if p <= 0.0:
                continue
            action = self._index_to_action(action_idx, legal, round_state)
            if action is None:
                continue
            action_value = self._calculate_action_utility(
                round_state, active, action, p, depth + 1
            )
            expected_value += p * action_value
            total_p += p

        if total_p <= 0.0:
            return self._leaf_value(round_state, active)
        return float(expected_value / total_p)
    
    def _train_networks(self):
        if len(self.replay_buffer) < self.batch_size:
            if self.training_mode and getattr(self, 'print_training', True):
                now = time.time()
                last = float(getattr(self, '_last_buffer_wait_print', 0.0))
                if now - last >= float(getattr(self, 'print_every_seconds', 30.0)):
                    print(f"[train] waiting for replay buffer: {len(self.replay_buffer)}/{self.batch_size}", flush=True)
                    self._last_buffer_wait_print = now
            return
        
        samples, indices, weights = self.replay_buffer.sample(self.batch_size)
        if samples is None:
            return
        
        states = []
        actions = []
        rewards = []
        next_states = []
        next_legal_masks = []
        dones = []
        returns = []
        advantages = []
        legal_masks = []
        old_log_probs = []
        raise_fractions = []
        players = []
        streets = []
        
        for sample in samples:
            states.append(sample['state'])
            actions.append(sample['action'])
            rewards.append(sample['reward'])
            next_states.append(sample['next_state'])
            next_legal_masks.append(sample.get('next_legal_mask', None))
            dones.append(sample['done'])
            returns.append(sample['return'])
            advantages.append(sample['advantage'])
            legal_masks.append(sample['legal_mask'])
            old_log_probs.append(sample.get('old_log_prob', 0.0))
            raise_fractions.append(sample.get('raise_fraction', None))
            players.append(int(sample.get('player', 0)))
            streets.append(int(sample.get('street', 0)))
        
        _encoded = None
        if _ncfr_bool_env('NCFR_TRAIN_ENCODER', True):
            try:
                sds = [s.get('state_data') for s in samples]
                if all(isinstance(sd, dict) and 'card_indices' in sd for sd in sds):
                    def _cat(key):
                        return torch.cat([sd[key].to(self.device) if sd[key].dim() > 1
                                          else sd[key].unsqueeze(0).to(self.device)
                                          for sd in sds], dim=0)
                    _encoded = self.state_encoder(_cat('card_indices'), _cat('positions'),
                                                  _cat('game_features'), _cat('attention_mask'))
                    if self.use_bf16:
                        _encoded = _encoded.to(dtype=torch.bfloat16)
            except Exception as _e:
                if not getattr(self, '_enc_warned', False):
                    self._enc_warned = True
                    print(f"[encoder] re-encode failed, using frozen reps: {_e}", flush=True)
                _encoded = None
        if _encoded is not None:
            try:
                nsds = [s.get('next_state_data') for s in samples]
                idx = [k for k, sd in enumerate(nsds)
                       if isinstance(sd, dict) and 'card_indices' in sd]
                if idx:
                    def _ncat(key):
                        return torch.cat([nsds[k][key].to(self.device)
                                          if nsds[k][key].dim() > 1
                                          else nsds[k][key].unsqueeze(0).to(self.device)
                                          for k in idx], dim=0)
                    with torch.no_grad():
                        _nenc = self.state_encoder(_ncat('card_indices'), _ncat('positions'),
                                                   _ncat('game_features'), _ncat('attention_mask'))
                    if self.use_bf16:
                        _nenc = _nenc.to(dtype=torch.bfloat16)
                    for j, k in enumerate(idx):
                        next_states[k] = _nenc[j]
            except Exception as _e:
                if not getattr(self, '_nenc_warned', False):
                    self._nenc_warned = True
                    print(f"[encoder] next-state re-encode failed, using stored reps: {_e}",
                          flush=True)

        if (_encoded is not None and _ncfr_bool_env('NCFR_TRAIN_ENCODER', True)
                and self._accum_window_start('encoder')):
            self.encoder_optimizer.zero_grad(set_to_none=True)

        states_tensor = _encoded if _encoded is not None else torch.stack(states).to(self.device)
        actions_tensor = torch.tensor(actions, device=self.device)
        rewards_tensor = torch.tensor(rewards, device=self.device)
        returns_tensor = torch.tensor(returns, device=self.device)
        advantages_tensor = torch.tensor(advantages, device=self.device)
        weights_tensor = torch.tensor(weights, device=self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, device=self.device)
        
        if self.learning_steps < self.stage2_steps:
            self._train_policy_network(states_tensor, actions_tensor,
                advantages_tensor, returns_tensor,
                legal_masks, old_log_probs_tensor, weights_tensor,
                raise_fractions)
        elif self.learning_steps < self.stage3_steps:
            self._train_policy_network(states_tensor, actions_tensor,
                advantages_tensor, returns_tensor,
                legal_masks, old_log_probs_tensor, weights_tensor,
                raise_fractions)
            self._train_q_network(states_tensor, actions_tensor, rewards_tensor, 
                next_states, dones, legal_masks, next_legal_masks, weights_tensor)
        else:
            self._train_policy_network(states_tensor, actions_tensor,
                advantages_tensor, returns_tensor,
                legal_masks, old_log_probs_tensor, weights_tensor,
                raise_fractions)
            self._train_q_network(states_tensor, actions_tensor, rewards_tensor, 
                next_states, dones, legal_masks, next_legal_masks, weights_tensor)
            if getattr(self, 'train_combiner', True):
                self._train_strategy_combiner(states_tensor, actions_tensor, advantages_tensor, legal_masks, streets)
            self._train_deep_mccfr_networks()

        if (_encoded is not None and _ncfr_bool_env('NCFR_TRAIN_ENCODER', True)
                and self._accum_ready('encoder')):
            torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), 1.0)
            self.encoder_optimizer.step()
            self.encoder_optimizer.zero_grad(set_to_none=True)

        if self._accum_ready('critic'):
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.baseline_network.parameters(), 1.0)
            self.value_optimizer.step()
            self.value_optimizer.zero_grad(set_to_none=True)

        self.learning_steps += 1
        if self.learning_steps % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _accum_window_start(self, key: str) -> bool:
        return self._accum_counts.get(key, 0) == 0

    def _accum_ready(self, key: str) -> bool:
        n = self._accum_counts.get(key, 0) + 1
        if n % self.accum_steps == 0:
            self._accum_counts[key] = 0
            return True
        self._accum_counts[key] = n
        return False

    def _train_q_network(self, states, actions, rewards, next_states, dones, legal_masks, next_legal_masks, weights):
        if self._accum_window_start('q'):
            self.q_optimizer.zero_grad(set_to_none=True)
        
        if self.use_bf16:
            states = states.to(dtype=torch.bfloat16)
            rewards = rewards.to(dtype=torch.bfloat16)
            weights = weights.to(dtype=torch.bfloat16)
        
        q_atoms = self.q_network(states)
        batch_indices = torch.arange(len(actions))
        current_q_atoms = q_atoms[batch_indices, actions]
        
        with torch.no_grad():
            target_atoms_list = []
            num_atoms = int(current_q_atoms.shape[-1])
            for i, next_state in enumerate(next_states):
                if next_state is not None and not dones[i]:
                    next_state_dev = next_state.to(self.device)
                    if self.use_bf16:
                        next_state_dev = next_state_dev.to(dtype=torch.bfloat16)
                    next_q_atoms = self.target_q_network(next_state_dev.unsqueeze(0)).squeeze(0)
                    next_q_mean = next_q_atoms.mean(dim=-1)
                    nmask = None
                    try:
                        nmask = next_legal_masks[i]
                    except Exception:
                        nmask = None
                    if nmask is not None:
                        try:
                            nmask_t = nmask.to(self.device).float().view(-1)
                            if nmask_t.numel() == next_q_mean.numel() and float(nmask_t.sum().item()) > 0:
                                next_q_mean = next_q_mean.masked_fill(nmask_t <= 0, -1e9)
                        except Exception:
                            pass
                    best_a = int(torch.argmax(next_q_mean).item())
                    targ_atoms = rewards[i].to(self.device).view(1) + self.gamma * next_q_atoms[best_a]
                    targ_atoms = torch.nan_to_num(targ_atoms, nan=0.0, posinf=800.0, neginf=-800.0)
                    target_atoms_list.append(targ_atoms.view(-1))
                else:
                    target_atoms_list.append(rewards[i].to(self.device).repeat(num_atoms))
            target_atoms = torch.stack(target_atoms_list, dim=0).to(self.device)
            if self.use_bf16:
                target_atoms = target_atoms.to(dtype=torch.bfloat16)
        
        tau = self.q_network.tau.to(self.device)
        diff = target_atoms - current_q_atoms
        loss = torch.where(diff > 0, tau * diff, (tau - 1) * diff)
        loss = loss.abs().mean(dim=-1)
        loss = (loss * weights).mean()

        _q_bad = bool(torch.isnan(loss).any() or torch.isinf(loss).any())
        if _q_bad or _ncfr_bool_env('NCFR_DEBUG_Q', False):
            print(f"[DEBUG] q_atoms nan/inf: {torch.isnan(q_atoms).any()}, {torch.isinf(q_atoms).any()}, mean={q_atoms.mean()}, min={q_atoms.min()}, max={q_atoms.max()}")
            print(f"[DEBUG] target_atoms nan/inf: {torch.isnan(target_atoms).any()}, {torch.isinf(target_atoms).any()}, mean={target_atoms.mean()}, min={target_atoms.min()}, max={target_atoms.max()}")
            print(f"[DEBUG] diff nan/inf: {torch.isnan(diff).any()}, {torch.isinf(diff).any()}")
            print(f"[DEBUG] Q loss raw: {loss}, has_nan={torch.isnan(q_atoms).any()}, has_inf={torch.isinf(q_atoms).any()}")
        loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=-100.0)
        loss = loss / self.accum_steps
        loss.backward(retain_graph=True)
        if self._accum_ready('q'):
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.q_optimizer.step()
            self.q_optimizer.zero_grad(set_to_none=True)
            
            if getattr(self, 'writer', None) is not None:
                self.writer.add_scalar('Q Loss', loss.item() * self.accum_steps, self.learning_steps)
            try:
                self._last_losses['q_loss'] = float(loss.item() * self.accum_steps)
            except Exception:
                pass
        
        td_errors = diff.abs().mean(dim=-1).detach().cpu()
        if td_errors.dtype == torch.bfloat16:
            td_errors = td_errors.float()
        td_errors = td_errors.numpy()
    
    def _train_policy_network(self, states, actions, advantages, returns, legal_masks, old_log_probs, weights, raise_fractions):
        if self._accum_window_start('policy'):
            self.policy_optimizer.zero_grad(set_to_none=True)
        
        if self.use_bf16:
            states = states.to(dtype=torch.bfloat16)
            advantages = advantages.to(dtype=torch.bfloat16)
            returns = returns.to(dtype=torch.bfloat16)
            weights = weights.to(dtype=torch.bfloat16)
            old_log_probs = old_log_probs.to(dtype=torch.bfloat16)
        
        legal_mask_tensor = None
        if legal_masks and legal_masks[0] is not None:
            try:
                legal_mask_tensor = torch.stack([torch.from_numpy(lm) if isinstance(lm, np.ndarray) else lm for lm in legal_masks]).to(self.device)
                if self.use_bf16:
                    legal_mask_tensor = legal_mask_tensor.to(dtype=torch.bfloat16)
            except Exception:
                legal_mask_tensor = None
        
        policy_output = self.policy_network(states, legal_mask_tensor, training=True)
        
        new_log_probs = policy_output['log_probs']
        batch_indices = torch.arange(len(actions))
        action_log_probs = new_log_probs[batch_indices, actions]
        
        joint_log_probs = action_log_probs.clone()
        for i in range(len(actions)):
            a = int(actions[i].item())
            frac = raise_fractions[i]
            if a >= 2 and frac is not None:
                bucket = a - 2
                alpha, beta = self._beta_params(states[i:i+1], bucket)
                dist = Beta(alpha, beta)
                frac_t = torch.tensor([float(frac)], dtype=torch.float32, device=self.device)
                if self.use_bf16:
                    frac_t = frac_t.to(dtype=torch.bfloat16)
                joint_log_probs[i] = joint_log_probs[i] + dist.log_prob(frac_t)[0]
        
        ratio = torch.exp(joint_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        policy_loss = (policy_loss * weights).mean()
        
        baseline = policy_output['baseline'].squeeze()
        _ret_detached = returns.detach()
        try:
            _batch_scale = float(_ret_detached.abs().float().mean().item())
        except Exception:
            _batch_scale = 1.0
        if not math.isfinite(_batch_scale) or _batch_scale <= 0.0:
            _batch_scale = 1.0
        _prev = float(getattr(self, '_value_scale', 0.0) or 0.0)
        self._value_scale = _batch_scale if _prev <= 0.0 else 0.99 * _prev + 0.01 * _batch_scale
        _scale = max(1.0, float(self._value_scale))
        value_loss = F.smooth_l1_loss(baseline / _scale, _ret_detached / _scale)
        
        probs = policy_output['action_probs']
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        total_loss = policy_loss + 0.5 * value_loss - 0.05 * entropy
        
        total_loss = total_loss / self.accum_steps
        total_loss.backward(retain_graph=True)
        if self._accum_ready('policy'):
            torch.nn.utils.clip_grad_norm_(self._actor_params, 0.5)
            self.policy_optimizer.step()
            self.policy_optimizer.zero_grad(set_to_none=True)
            
            if getattr(self, 'writer', None) is not None:
                self.writer.add_scalar('Policy Loss', policy_loss.item() * self.accum_steps, self.learning_steps)
                self.writer.add_scalar('Value Loss', value_loss.item(), self.learning_steps)
                self.writer.add_scalar('Entropy', entropy.item(), self.learning_steps)
            
            try:
                self._last_losses['policy_loss'] = float(policy_loss.item() * self.accum_steps)
                self._last_losses['value_loss'] = float(value_loss.item())
                self._last_losses['entropy'] = float(entropy.item())
            except Exception:
                pass

    def _train_strategy_combiner(self, states: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor, legal_masks, streets: List[int]) -> None:
        if states.numel() == 0:
            return
        
        if self.use_bf16:
            states = states.to(dtype=torch.bfloat16)
            advantages = advantages.to(dtype=torch.bfloat16)
        
        adv = advantages.detach().float().clamp_min(0.0)
        if float(adv.sum().item()) <= 0:
            return
        
        try:
            legal = torch.stack(legal_masks).to(self.device)
            if self.use_bf16:
                legal = legal.to(dtype=torch.bfloat16)
        except Exception:
            legal = None
        
        with torch.no_grad():
            states_detached = states.detach()
            if self.use_bf16:
                states_detached = states_detached.to(dtype=torch.bfloat16)
            
            pol_out = self.policy_network(states_detached, legal, training=False)
            pi_pol = pol_out['action_probs']
            if legal is not None:
                pi_pol = pi_pol * legal
                pi_pol = pi_pol / pi_pol.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            q_values = self.q_network.get_q_values(states_detached)
            q_logits = q_values.masked_fill(legal == 0, -1e9) if legal is not None else q_values
            pi_q = F.softmax(q_logits, dim=-1)
            if legal is not None:
                pi_q = pi_q * legal
                pi_q = pi_q / pi_q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            pi_cfr = self.avg_strategy_network(states_detached, legal)
            if legal is not None:
                pi_cfr = pi_cfr * legal
                pi_cfr = pi_cfr / pi_cfr.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        
        if self._accum_window_start('combiner'):
            self.combiner_optimizer.zero_grad(set_to_none=True)
        
        w = self.strategy_combiner(states)
        try:
            s = torch.tensor([int(max(0, min(6, int(x)))) for x in streets], device=self.device, dtype=torch.long)
            prior_logits = self.combiner_street_prior(s)
            prior = F.softmax(prior_logits, dim=-1)
        except Exception:
            prior = torch.ones_like(w) / 3.0
        
        w = w * prior
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        
        pi_mix = w[:, 0:1] * pi_q + w[:, 1:2] * pi_pol + w[:, 2:3] * pi_cfr
        if legal is not None:
            pi_mix = pi_mix * legal
        pi_mix = pi_mix / pi_mix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pi_mix = pi_mix.clamp_min(1e-12)
        
        batch_idx = torch.arange(actions.shape[0], device=self.device)
        logp = torch.log(pi_mix[batch_idx, actions])
        loss = -((adv / adv.mean().clamp_min(1e-6)) * logp).mean()
        
        try:
            ent = -(w * torch.log(w.clamp_min(1e-12))).sum(dim=-1).mean()
            loss = loss - float(getattr(self, 'combiner_entropy_coef', 0.0)) * ent
        except Exception:
            pass
        try:
            kl = (w * (torch.log(w.clamp_min(1e-12)) - torch.log(prior.clamp_min(1e-12)))).sum(dim=-1).mean()
            loss = loss + float(getattr(self, 'combiner_prior_kl_coef', 0.0)) * kl
        except Exception:
            pass
        
        loss = loss / self.accum_steps
        loss.backward(retain_graph=True)
        if self._accum_ready('combiner'):
            torch.nn.utils.clip_grad_norm_(list(self.strategy_combiner.parameters()) + list(self.combiner_street_prior.parameters()), 1.0)
            self.combiner_optimizer.step()
            self.combiner_optimizer.zero_grad(set_to_none=True)
            
            if getattr(self, 'writer', None) is not None:
                self.writer.add_scalar('Combiner Loss', loss.item() * self.accum_steps, self.learning_steps)
            try:
                self._last_losses['comb_loss'] = float(loss.item() * self.accum_steps)
            except Exception:
                pass
    
    def _train_deep_mccfr_networks(self):
        if len(self.mccfr_advantage_buffer) < max(1024, self.mccfr_batch_size):
            return
        if len(self.mccfr_strategy_buffer) < max(1024, self.mccfr_batch_size):
            return
    
        batch = self.mccfr_advantage_buffer.sample(self.mccfr_batch_size)
        by_pid = {0: [], 1: []}
        for b in batch:
            by_pid[int(b.get('player', 0))].append(b)
    
        for pid, items in by_pid.items():
            if not items:
                continue
            states = torch.stack([b['state'] for b in items]).to(self.device)
            legal = torch.stack([b['legal_mask'] for b in items]).to(self.device)
            target_regrets = torch.stack([b['target_regrets'] for b in items]).to(self.device)
            
            if self.use_bf16:
                states = states.to(dtype=torch.bfloat16)
                target_regrets = target_regrets.to(dtype=torch.bfloat16)
                legal = legal.to(dtype=torch.bfloat16)
            
            opt = self.cfr_optimizers[pid]
            net = self.cfr_networks[pid]
            if self._accum_window_start(f'mccfr_adv_{pid}'):
                opt.zero_grad(set_to_none=True)
            
            out = net(states)
            pred = out['regret_pred']
            pred = pred.masked_fill(legal == 0, 0.0)
            target = target_regrets.masked_fill(legal == 0, 0.0)
            adv_loss = F.mse_loss(pred, target)
            
            adv_loss = adv_loss / self.accum_steps
            adv_loss.backward()
            if self._accum_ready(f'mccfr_adv_{pid}'):
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                
                if getattr(self, 'writer', None) is not None:
                    self.writer.add_scalar(f'MCCFR Adv Loss (pid {pid})', adv_loss.item() * self.accum_steps, self.learning_steps)
                try:
                    self._last_losses[f'mccfr_adv_loss_pid{pid}'] = float(adv_loss.item() * self.accum_steps)
                except Exception:
                    pass
    
        if len(self.mccfr_strategy_buffer) >= max(1024, self.mccfr_batch_size):
            batch = self.mccfr_strategy_buffer.sample(self.mccfr_batch_size)
            states = torch.stack([b['state'] for b in batch]).to(self.device)
            legal = torch.stack([b['legal_mask'] for b in batch]).to(self.device)
            target_pi = torch.stack([b['target_strategy'] for b in batch]).to(self.device)
            weights = torch.tensor([float(b.get('weight', 1.0)) for b in batch], device=self.device)
            
            if self.use_bf16:
                states = states.to(dtype=torch.bfloat16)
                target_pi = target_pi.to(dtype=torch.bfloat16)
                legal = legal.to(dtype=torch.bfloat16)
                weights = weights.to(dtype=torch.bfloat16)
            
            try:
                clip_mult = float(getattr(self, 'mccfr_weight_clip', 10.0))
                med = torch.median(weights)
                if torch.isfinite(med) and med.item() > 0:
                    weights = torch.clamp(weights, max=med * clip_mult)
                mean = weights.mean().clamp_min(1e-6)
                weights = weights / mean
            except Exception:
                pass
            
            if self._accum_window_start('mccfr_strat'):
                self.avg_strategy_optimizer.zero_grad(set_to_none=True)
            pred_pi = self.avg_strategy_network(states, legal)
            pred_pi = pred_pi.clamp_min(1e-12)
            ce = -(target_pi * torch.log(pred_pi)).sum(dim=-1)
            strat_loss = (ce * weights).mean()
            
            pred_pi = torch.nan_to_num(pred_pi, nan=1e-12)
            pred_pi = pred_pi.clamp_min(1e-12)
            strat_loss = torch.nan_to_num(strat_loss, nan=0.0)
            
            strat_loss = strat_loss / self.accum_steps
            strat_loss.backward()
            if self._accum_ready('mccfr_strat'):
                torch.nn.utils.clip_grad_norm_(self.avg_strategy_network.parameters(), 1.0)
                self.avg_strategy_optimizer.step()
                self.avg_strategy_optimizer.zero_grad(set_to_none=True)
                
                if getattr(self, 'writer', None) is not None:
                    self.writer.add_scalar('MCCFR Strat Loss', strat_loss.item() * self.accum_steps, self.learning_steps)
                try:
                    self._last_losses['mccfr_strat_loss'] = float(strat_loss.item() * self.accum_steps)
                except Exception:
                    pass
    
        if len(self.mccfr_value_buffer) >= max(1024, getattr(self, 'mccfr_value_batch_size', self.mccfr_batch_size)):
            batch = self.mccfr_value_buffer.sample(int(getattr(self, 'mccfr_value_batch_size', self.mccfr_batch_size)))
            states = torch.stack([b['state'] for b in batch]).to(self.device)
            targets = torch.stack([b['target_value'] for b in batch]).to(self.device).view(-1)
            vweights = torch.tensor([float(b.get('weight', 1.0)) for b in batch], device=self.device)
            
            if self.use_bf16:
                states = states.to(dtype=torch.bfloat16)
                targets = targets.to(dtype=torch.bfloat16)
                vweights = vweights.to(dtype=torch.bfloat16)
            
            try:
                clip_mult = float(getattr(self, 'mccfr_weight_clip', 10.0))
                med = torch.median(vweights)
                if torch.isfinite(med) and med.item() > 0:
                    vweights = torch.clamp(vweights, max=med * clip_mult)
                mean = vweights.mean().clamp_min(1e-6)
                vweights = vweights / mean
            except Exception:
                pass
            
            out = self.policy_network(states, legal_mask=None, training=False)
            pred_v = out['baseline'].view(-1)
            _vt = targets.detach()
            try:
                _bs = float(_vt.abs().float().mean().item())
            except Exception:
                _bs = 1.0
            if not math.isfinite(_bs) or _bs <= 0.0:
                _bs = 1.0
            _prev = float(getattr(self, '_value_scale', 0.0) or 0.0)
            self._value_scale = _bs if _prev <= 0.0 else 0.99 * _prev + 0.01 * _bs
            _vs = max(1.0, float(self._value_scale))
            vloss = F.smooth_l1_loss(pred_v / _vs, _vt / _vs, reduction='none') * vweights
            vloss = vloss.mean()
            
            vloss = vloss / self.accum_steps
            vloss.backward()
            
            if getattr(self, 'writer', None) is not None:
                self.writer.add_scalar('MCCFR Value Loss', vloss.item() * self.accum_steps, self.learning_steps)
            try:
                self._last_losses['mccfr_value_loss'] = float(vloss.item() * self.accum_steps)
            except Exception:
                pass
            
    def _deep_mccfr_update_from_observed_hand(self, terminal_state: TerminalState):
        observed = terminal_state.previous_state
        if observed is None or not hasattr(observed, 'hands') or not hasattr(observed, 'board'):
            return
    
        def _as_str_list(cards: Any) -> List[str]:
            try:
                seq = list(cards)
            except Exception:
                return []
            out: List[str] = []
            for c in seq:
                try:
                    out.append(str(c))
                except Exception:
                    continue
            return out
    
        try:
            obs_hands = [_as_str_list(observed.hands[0]), _as_str_list(observed.hands[1])]
        except Exception:
            return
        if len(obs_hands[0]) == 0 or len(obs_hands[1]) == 0:
            return
        obs_board = _as_str_list(getattr(observed, 'board', []) or [])
        if len(obs_board) < 2:
            return
    
        if len(obs_hands[0]) == 3 and len(obs_hands[1]) == 3:
            pre_hands = [list(obs_hands[0]), list(obs_hands[1])]
        elif len(obs_hands[0]) == 2 and len(obs_hands[1]) == 2 and len(obs_board) >= 4:
            pre_hands = [list(obs_hands[0]) + [obs_board[3]], list(obs_hands[1]) + [obs_board[2]]]
        else:
            return
    
        ranks = "23456789TJQKA"
        suits = "cdhs"
        full_deck = [r + s for r in ranks for s in suits]
        dead = set(pre_hands[0] + pre_hands[1])
        remaining = [c for c in full_deck if c not in dead]
        if len(remaining) < 4:
            return
    
        def _best5_value(cards: List[pkrbot.Card]) -> int:
            if len(cards) <= 5:
                return int(pkrbot.evaluate(cards))
            best = None
            for combo in itertools.combinations(cards, 5):
                v = int(pkrbot.evaluate(list(combo)))
                if best is None or v > best:
                    best = v
            return int(best) if best is not None else 0
    
        class _TrainRoundState:
            __slots__ = ('button', 'street', 'pips', 'stacks', 'hands', 'board', 'chance', 'previous_state')
            def __init__(self, button, street, pips, stacks, hands, board, chance, previous_state=None):
                self.button = int(button)
                self.street = int(street)
                self.pips = list(pips)
                self.stacks = list(stacks)
                self.hands = [list(hands[0]), list(hands[1])]
                self.board = list(board)
                self.chance = list(chance)
                self.previous_state = previous_state
    
            def legal_actions(self):
                active = self.button % 2
                continue_cost = self.pips[1 - active] - self.pips[active]
                if self.street in (2, 3):
                    return {DiscardAction} if active != (self.street % 2) else {CheckAction}
                if continue_cost == 0:
                    bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
                    return {CheckAction, FoldAction} if bets_forbidden else {CheckAction, RaiseAction, FoldAction}
                raises_forbidden = (continue_cost == self.stacks[active] or self.stacks[1 - active] == 0)
                return {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}
    
            def raise_bounds(self):
                active = self.button % 2
                continue_cost = self.pips[1 - active] - self.pips[active]
                max_contribution = min(self.stacks[active], self.stacks[1 - active] + continue_cost)
                min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND))
                return (self.pips[active] + min_contribution, self.pips[active] + max_contribution)
    
            def _get_delta(self, winner_index: int) -> int:
                if winner_index == 2:
                    return 0
                if winner_index == 0:
                    return int(STARTING_STACK - self.stacks[1])
                return int(self.stacks[0] - STARTING_STACK)
    
            def _showdown(self):
                cards0 = [pkrbot.Card(c) for c in (self.board + self.hands[0])]
                cards1 = [pkrbot.Card(c) for c in (self.board + self.hands[1])]
                v0 = _best5_value(cards0)
                v1 = _best5_value(cards1)
                if v0 > v1:
                    d0 = self._get_delta(0)
                elif v1 > v0:
                    d0 = self._get_delta(1)
                else:
                    d0 = self._get_delta(2)
                return TerminalState([int(d0), int(-d0)], self)
    
            def proceed_street(self):
                if self.street == 6:
                    return self._showdown()
                if self.street == 0:
                    if len(self.board) < 2:
                        self.board.extend(self.chance[:2])
                    return _TrainRoundState(1, 2, [0, 0], self.stacks, self.hands, self.board, self.chance, self)
                if self.street == 2:
                    return _TrainRoundState(0, 3, [0, 0], self.stacks, self.hands, self.board, self.chance, self)
                if self.street == 3:
                    return _TrainRoundState(1, 4, [0, 0], self.stacks, self.hands, self.board, self.chance, self)
                if self.street == 4:
                    if len(self.board) < 5:
                        self.board.append(self.chance[2])
                    return _TrainRoundState(1, 5, [0, 0], self.stacks, self.hands, self.board, self.chance, self)
                if self.street == 5:
                    if len(self.board) < 6:
                        self.board.append(self.chance[3])
                    return _TrainRoundState(1, 6, [0, 0], self.stacks, self.hands, self.board, self.chance, self)
                return _TrainRoundState(1, self.street, [0, 0], self.stacks, self.hands, self.board, self.chance, self)
    
            def proceed(self, action):
                active = self.button % 2
                if isinstance(action, DiscardAction):
                    if 0 <= int(action.card) < len(self.hands[active]):
                        self.board.append(self.hands[active].pop(int(action.card)))
                    return _TrainRoundState(1 - active, self.street, self.pips, self.stacks, self.hands, self.board, self.chance, self)
                if isinstance(action, FoldAction):
                    d0 = self._get_delta(1 - active)
                    return TerminalState([int(d0), int(-d0)], self)
                if isinstance(action, CallAction):
                    if self.street == 0 and self.button == 0:
                        return _TrainRoundState(
                            1,
                            0,
                            [BIG_BLIND, BIG_BLIND],
                            [STARTING_STACK - BIG_BLIND, STARTING_STACK - BIG_BLIND],
                            self.hands,
                            self.board,
                            self.chance,
                            self,
                        )
                    new_pips = list(self.pips)
                    new_stacks = list(self.stacks)
                    contribution = new_pips[1 - active] - new_pips[active]
                    new_stacks[active] -= contribution
                    new_pips[active] += contribution
                    state = _TrainRoundState(self.button + 1, self.street, new_pips, new_stacks, self.hands, self.board, self.chance, self)
                    return state.proceed_street()
                if isinstance(action, CheckAction):
                    if self.street in (2, 3) or (self.street == 0 and self.button > 0) or self.button > 1:
                        return self.proceed_street()
                    return _TrainRoundState(self.button + 1, self.street, self.pips, self.stacks, self.hands, self.board, self.chance, self)
                new_pips = list(self.pips)
                new_stacks = list(self.stacks)
                contribution = int(action.amount) - new_pips[active]
                new_stacks[active] -= contribution
                new_pips[active] += contribution
                return _TrainRoundState(self.button + 1, self.street, new_pips, new_stacks, self.hands, self.board, self.chance, self)
    
        rep_cache: Dict[Any, torch.Tensor] = {}
        def state_rep_for(state: _TrainRoundState, player: int) -> Optional[torch.Tensor]:
            try:
                key = (
                    player, state.street, state.button,
                    int(state.pips[0]), int(state.pips[1]),
                    int(state.stacks[0]), int(state.stacks[1]),
                    tuple(state.hands[player]), tuple(state.board)
                )
            except Exception:
                return None
            if key in rep_cache:
                return rep_cache[key]
            try:
                sd = self.encode_state_tensor(state, player)
                with torch.inference_mode():
                    rep = self.state_encoder(sd['card_indices'], sd['positions'], sd['game_features'], sd['attention_mask'])
                rep_cache[key] = rep
                return rep
            except Exception:
                return None
    
        def legal_mask_for(state: _TrainRoundState, acting: int) -> np.ndarray:
            legal = state.legal_actions()
            mask = np.zeros(int(self.num_actions), dtype=np.float32)
            if FoldAction in legal:
                mask[0] = 1.0
            if CallAction in legal or CheckAction in legal:
                mask[1] = 1.0
            if RaiseAction in legal:
                try:
                    mn, mx = state.raise_bounds()
                    if mx >= mn and mx > 0:
                        mask[2:6] = 1.0
                except Exception:
                    pass
            if DiscardAction in legal:
                n = min(MAX_HOLE_CARDS, len(state.hands[acting]))
                for i in range(int(n)):
                    mask[6 + i] = 1.0
            return mask
    
        def index_to_action(idx: int, state: _TrainRoundState, player: int):
            rep = state_rep_for(state, player)
            if rep is None:
                return None
            return self._convert_to_engine_action(idx, state.legal_actions(), state, rep, raise_fraction=None)
    
        def strategy_for(state: _TrainRoundState, player: int) -> np.ndarray:
            rep = state_rep_for(state, player)
            if rep is None:
                return np.ones(int(self.num_actions), dtype=np.float32) / float(self.num_actions)
            mask_np = legal_mask_for(state, player)
            mask = torch.from_numpy(mask_np.astype(np.float32, copy=False)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                out_rm = self.cfr_networks[int(player)](rep)
                sigma_rm = self.cfr_networks[int(player)].compute_regret_matching_strategy(out_rm['regret_pred'], mask)
                sigma_avg = self.avg_strategy_network(rep, mask)
                upd = float(getattr(self, 'mccfr_updates_seen', 0))
                denom = float(max(1, int(getattr(self, 'mccfr_avg_mix_anneal_updates', 1))))
                alpha = max(0.0, min(1.0, upd / denom))
                strat = (1.0 - alpha) * sigma_rm + alpha * sigma_avg
            return strat.detach().float().cpu().numpy()[0]
    
        def sample_action(mask_np: np.ndarray, probs_np: np.ndarray) -> int:
            p = probs_np * mask_np
            s = float(p.sum())
            if s <= 0:
                for i in range(int(self.num_actions)):
                    if mask_np[i] > 0:
                        return i
                return 1
            p = p / s
            return int(np.random.choice(int(self.num_actions), p=p))
    
        def traverse(state, traverser: int, pi0: float, pi1: float) -> float:
            if isinstance(state, TerminalState):
                return float(state.deltas[traverser])
            current = state.button % 2
            mask_np = legal_mask_for(state, current)
            sigma = strategy_for(state, current)
            sigma = sigma * mask_np
            if sigma.sum() > 0:
                sigma = sigma / sigma.sum()
            else:
                sigma = mask_np / max(mask_np.sum(), 1.0)
            rep = state_rep_for(state, current)
            if rep is not None:
                w = pi1 if current == 0 else pi0
                it = int(getattr(self, 'mccfr_updates_seen', 0))
                self.mccfr_strategy_buffer.add({
                    'state': rep.detach().cpu().squeeze(0),
                    'legal_mask': torch.tensor(mask_np, dtype=torch.float32),
                    'target_strategy': torch.tensor(sigma, dtype=torch.float32),
                    'weight': float(w) * float(max(1, it)),
                    'reach': float(w),
                    'iteration': int(it),
                })
            if current != traverser:
                a = sample_action(mask_np, sigma)
                act = index_to_action(a, state, current)
                if act is None:
                    return 0.0
                nxt = state.proceed(act)
                if current == 0:
                    return traverse(nxt, traverser, pi0 * float(sigma[a]), pi1)
                return traverse(nxt, traverser, pi0, pi1 * float(sigma[a]))
            util = np.zeros(int(self.num_actions), dtype=np.float32)
            for a in range(int(self.num_actions)):
                if mask_np[a] == 0:
                    continue
                act = index_to_action(a, state, current)
                if act is None:
                    continue
                nxt = state.proceed(act)
                if current == 0:
                    util[a] = traverse(nxt, traverser, pi0 * float(sigma[a]), pi1)
                else:
                    util[a] = traverse(nxt, traverser, pi0, pi1 * float(sigma[a]))
            node_util = float((util * sigma).sum())
            regrets = util - node_util
            if rep is not None:
                reach = float(pi0 * pi1)
                self.mccfr_value_buffer.add({
                    'state': rep.detach().cpu().squeeze(0),
                    'target_value': torch.tensor(float(node_util), dtype=torch.float32),
                    'weight': float(reach) * float(max(1, int(getattr(self, 'mccfr_updates_seen', 0)))),
                    'player': int(traverser),
                })
                self.mccfr_advantage_buffer.add({
                    'state': rep.detach().cpu().squeeze(0),
                    'legal_mask': torch.tensor(mask_np, dtype=torch.float32),
                    'target_regrets': torch.tensor(regrets, dtype=torch.float32),
                    'player': int(traverser),
                })
            return node_util
    
        chance_k = int(max(1, getattr(self, 'mccfr_chance_samples_per_hand', 1)))
        for _ in range(int(self.mccfr_traversals_per_hand)):
            for _ in range(int(chance_k)):
                chance = random.sample(remaining, 4)
                root = _TrainRoundState(
                    0,
                    0,
                    [SMALL_BLIND, BIG_BLIND],
                    [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND],
                    pre_hands,
                    [],
                    chance,
                    None,
                )
                traverse(root, traverser=0, pi0=1.0, pi1=1.0)
                traverse(root, traverser=1, pi0=1.0, pi1=1.0)
    
                try:
                    extra = int(getattr(self, 'mccfr_discard_focus_samples', 0))
                except Exception:
                    extra = 0
                if extra > 0:
                    flop_board = list(chance[:2])
                    stacks = [STARTING_STACK - BIG_BLIND, STARTING_STACK - BIG_BLIND]
                    for _ in range(int(extra)):
                        root_d = _TrainRoundState(
                            1,
                            2,
                            [0, 0],
                            stacks,
                            pre_hands,
                            flop_board,
                            chance,
                            None,
                        )
                        traverse(root_d, traverser=0, pi0=1.0, pi1=1.0)
                        traverse(root_d, traverser=1, pi0=1.0, pi1=1.0)
                    
    def load_models(self):
        model_dir = _ncfr_model_dir()
        if os.path.exists(model_dir):
            try:
                checkpoint = torch.load(
                    os.path.join(model_dir, 'checkpoint.pt'),
                    map_location=self.device
                )

                expected_version = 1
                version = checkpoint.get('version', None)
                if version != expected_version:
                    print(f"[checkpoint] WARNING: Checkpoint version mismatch (found {version}, expected {expected_version})")

                metrics = checkpoint.get('validation_metrics', None)
                if metrics is not None:
                    print(f"[checkpoint] Validation metrics: {metrics}")
                    import math
                    for k, v in metrics.items():
                        if not isinstance(v, float) or not math.isfinite(v):
                            print(f"[checkpoint] WARNING: Metric {k} is not finite (value={v})")

                self.state_encoder.load_state_dict(checkpoint['state_encoder'])
                self.q_network.load_state_dict(checkpoint['q_network'])
                self.target_q_network.load_state_dict(checkpoint['target_q_network'])
                self.policy_network.load_state_dict(checkpoint['policy_network'])
                if 'cfr_networks' in checkpoint:
                    nets = checkpoint['cfr_networks']
                    if isinstance(nets, (list, tuple)) and len(nets) >= 2:
                        self.cfr_networks[0].load_state_dict(nets[0])
                        self.cfr_networks[1].load_state_dict(nets[1])
                    elif isinstance(nets, (list, tuple)) and len(nets) == 1:
                        self.cfr_networks[0].load_state_dict(nets[0])
                        self.cfr_networks[1].load_state_dict(nets[0])
                elif 'cfr_network' in checkpoint:
                    self.cfr_networks[0].load_state_dict(checkpoint['cfr_network'])
                    self.cfr_networks[1].load_state_dict(checkpoint['cfr_network'])
                if 'avg_strategy_network' in checkpoint:
                    self.avg_strategy_network.load_state_dict(checkpoint['avg_strategy_network'])

                if 'strategy_combiner' in checkpoint:
                    try:
                        self.strategy_combiner.load_state_dict(checkpoint['strategy_combiner'])
                    except Exception:
                        pass
                if 'combiner_street_prior' in checkpoint:
                    try:
                        self.combiner_street_prior.load_state_dict(checkpoint['combiner_street_prior'])
                    except Exception:
                        pass
                if 'raise_sizing_head' in checkpoint:
                    try:
                        self.raise_sizing_head.load_state_dict(checkpoint['raise_sizing_head'])
                    except Exception:
                        pass
                
                if 'optimizers' in checkpoint:
                    self.encoder_optimizer.load_state_dict(checkpoint['optimizers']['encoder'])
                    self.q_optimizer.load_state_dict(checkpoint['optimizers']['q'])
                    self.policy_optimizer.load_state_dict(checkpoint['optimizers']['policy'])
                    if 'value' in checkpoint['optimizers']:
                        self.value_optimizer.load_state_dict(checkpoint['optimizers']['value'])
                    if 'cfr_0' in checkpoint['optimizers'] and 'cfr_1' in checkpoint['optimizers']:
                        self.cfr_optimizers[0].load_state_dict(checkpoint['optimizers']['cfr_0'])
                        self.cfr_optimizers[1].load_state_dict(checkpoint['optimizers']['cfr_1'])
                    elif 'cfr' in checkpoint['optimizers']:
                        self.cfr_optimizers[0].load_state_dict(checkpoint['optimizers']['cfr'])
                        self.cfr_optimizers[1].load_state_dict(checkpoint['optimizers']['cfr'])
                    if 'avg_strategy' in checkpoint['optimizers']:
                        self.avg_strategy_optimizer.load_state_dict(checkpoint['optimizers']['avg_strategy'])
                    if 'comb' in checkpoint['optimizers']:
                        try:
                            self.combiner_optimizer.load_state_dict(checkpoint['optimizers']['comb'])
                        except Exception:
                            pass
                
                if 'training_state' in checkpoint:
                    ts = checkpoint['training_state']
                    if not os.environ.get('NCFR_EPSILON', '').strip():
                        self.epsilon = ts.get('epsilon', self.epsilon)
                    self.learning_steps = ts.get('learning_steps', 0)
                    self.mccfr_updates_seen = int(ts.get('mccfr_updates_seen', getattr(self, 'mccfr_updates_seen', 0)))
                    for _k in ('stage2_steps', 'stage3_steps'):
                        _env = 'NCFR_' + _k.replace('_steps', '').upper() + '_STEPS'
                        if _k in ts and not os.environ.get(_env, '').strip():
                            setattr(self, _k, int(ts[_k]))
                
                print(f"Loaded models from {model_dir}")
                print(f"Learning steps: {self.learning_steps}")
                
            except Exception as e:
                print(f"Error loading models: {e}")
                print("Starting with fresh initialization")
    
    def save_models(self, snapshot_round: Optional[int] = None):
        model_dir = _ncfr_model_dir()
        os.makedirs(model_dir, exist_ok=True)

        checkpoint_version = 1
        validation_metrics = {
            'policy_loss': float(getattr(self, '_last_losses', {}).get('policy_loss', 0.0)),
            'q_loss': float(getattr(self, '_last_losses', {}).get('q_loss', 0.0)),
            'value_loss': float(getattr(self, '_last_losses', {}).get('value_loss', 0.0)),
            'entropy': float(getattr(self, '_last_losses', {}).get('entropy', 0.0)),
            'comb_loss': float(getattr(self, '_last_losses', {}).get('comb_loss', 0.0)),
        }

        checkpoint = {
            'version': checkpoint_version,
            'validation_metrics': validation_metrics,
            'state_encoder': self.state_encoder.state_dict(),
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'raise_sizing_head': self.raise_sizing_head.state_dict(),
            'cfr_networks': [self.cfr_networks[0].state_dict(), self.cfr_networks[1].state_dict()],
            'avg_strategy_network': self.avg_strategy_network.state_dict(),
            'strategy_combiner': self.strategy_combiner.state_dict(),
            'combiner_street_prior': self.combiner_street_prior.state_dict(),
            'optimizers': {
                'encoder': self.encoder_optimizer.state_dict(),
                'q': self.q_optimizer.state_dict(),
                'policy': self.policy_optimizer.state_dict(),
                'value': self.value_optimizer.state_dict(),
                'cfr_0': self.cfr_optimizers[0].state_dict(),
                'cfr_1': self.cfr_optimizers[1].state_dict(),
                'avg_strategy': self.avg_strategy_optimizer.state_dict(),
                'comb': self.combiner_optimizer.state_dict(),
            },
            'training_state': {
                'epsilon': self.epsilon,
                'learning_steps': self.learning_steps,
                'mccfr_updates_seen': int(getattr(self, 'mccfr_updates_seen', 0)),
                'replay_buffer_size': len(self.replay_buffer),
                'stage2_steps': int(getattr(self, 'stage2_steps', 0)),
                'stage3_steps': int(getattr(self, 'stage3_steps', 0))
            }
        }

        _ncfr_torch_save(checkpoint, os.path.join(model_dir, 'checkpoint.pt'))

        if snapshot_round is not None:
            try:
                snap_dir = os.path.join(model_dir, 'snapshots')
                os.makedirs(snap_dir, exist_ok=True)
                snap_path = os.path.join(snap_dir, f'checkpoint_v{checkpoint_version}_{int(snapshot_round)}.pt')
                _ncfr_torch_save(checkpoint, snap_path)

                keep = int(os.environ.get('NCFR_SNAPSHOT_KEEP', '3') or 3)
                if keep >= 0:
                    snaps = [f for f in os.listdir(snap_dir) if f.endswith('.pt')]
                    def _snap_key(name):
                        try:
                            return int(name.rsplit('_', 1)[1].split('.')[0])
                        except (IndexError, ValueError):
                            return -1
                    for stale in sorted(snaps, key=_snap_key)[:-keep or None]:
                        try:
                            os.remove(os.path.join(snap_dir, stale))
                        except OSError:
                            pass
            except Exception:
                pass

        if self.training_mode:
            if getattr(self, 'use_deep_mccfr', False):
                print(
                    f"[save] {model_dir} | deep_mccfr updates={int(getattr(self, 'mccfr_updates_seen', 0))}"
                    f" adv={len(getattr(self, 'mccfr_advantage_buffer', []))}"
                    f" strat={len(getattr(self, 'mccfr_strategy_buffer', []))}"
                    f" value={len(getattr(self, 'mccfr_value_buffer', []))}"
                    f" | replay={len(self.replay_buffer)}",
                    flush=True,
                )
            else:
                print(f"[save] {model_dir}"
                      f" | replay={len(self.replay_buffer)}", flush=True)

if __name__ == '__main__':
    bot = CompleteNeuralCFRBot()
    bot.training_mode = _ncfr_bool_env('NCFR_TRAINING', False)
    if not bot.training_mode:
        bot._sync_model_modes()
    POKERBOT_TRAINING = 1
    if bot.training_mode:
        import signal
        def save_on_exit(signum, frame):
            print("\nSaving models before exit...")
            bot.save_models()
            exit(0)
        
        signal.signal(signal.SIGINT, save_on_exit)
    
    run_bot(bot, parse_args())
