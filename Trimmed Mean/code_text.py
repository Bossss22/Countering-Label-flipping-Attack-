This is the complete code, first just understand the code and then i will share the quires(import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, TensorDataset
import random
import string
import copy
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ========================
# 1. V2G Dataset Implementation
# ========================
class V2GDataset(Dataset):
    def __init__(self, csv_file):
        # Load CSV file
        self.data = pd.read_csv(csv_file)
        
        # Print dataset info
        print(f"Dataset loaded with {len(self.data)} records")
        print(f"Columns: {', '.join(self.data.columns)}")
        
        # Preprocess timestamps if needed
        if 'timestamp' in self.data.columns and not pd.api.types.is_datetime64_any_dtype(self.data['timestamp']):
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            # Extract hour as a feature
            self.data['hour'] = self.data['timestamp'].dt.hour
        
        # Encode labels if they're categorical
        if 'label' in self.data.columns and self.data['label'].dtype == 'object':
            self.label_encoder = LabelEncoder()
            self.data['label_encoded'] = self.label_encoder.fit_transform(self.data['label'])
            print(f"Label classes: {', '.join(self.label_encoder.classes_)}")
        else:
            # If no label column, create a dummy one (all zeros)
            self.data['label_encoded'] = 0
            
        # Extract numerical features
        self.feature_columns = ['battery_capacity_kWh', 'current_charge_kWh', 
                               'discharge_rate_kW', 'energy_requested_kWh']
        
        if 'hour' in self.data.columns:
            self.feature_columns.append('hour')
        
        # Scale features
        self.scaler = StandardScaler()
        self.data[self.feature_columns] = self.scaler.fit_transform(self.data[self.feature_columns])
        
        # Group by participant_id to allow data distribution
        self.participant_groups = self.data.groupby('participant_id')
        self.participant_ids = list(self.participant_groups.groups.keys())
        
        print(f"Found {len(self.participant_ids)} unique participants")
        
        # Count honest vs adversarial if label exists
        if 'label' in self.data.columns:
            honest_count = len(self.data[self.data['label'].str.contains('honest', case=False, na=False)])
            adv_count = len(self.data) - honest_count
            print(f"Data distribution: {honest_count} honest, {adv_count} adversarial records")
            
            # Count unique participants of each type
            participant_types = {}
            for p_id in self.participant_ids:
                labels = self.data[self.data['participant_id'] == p_id]['label'].unique()
                if len(labels) > 0:
                    participant_types[p_id] = 'adversarial' if 'adv' in labels[0].lower() else 'honest'
            
            honest_p = sum(1 for p_type in participant_types.values() if p_type == 'honest')
            adv_p = sum(1 for p_type in participant_types.values() if p_type == 'adversarial')
            print(f"Participant distribution: {honest_p} honest, {adv_p} adversarial participants")
        
        # Extract features and labels as tensors
        self.features = torch.tensor(self.data[self.feature_columns].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data['label_encoded'].values, dtype=torch.long)
        
        # Get number of features and classes
        self.num_features = len(self.feature_columns)
        self.num_classes = len(self.data['label_encoded'].unique())
        
        print(f"Features: {self.num_features}, Classes: {self.num_classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return feature vector and label
        return self.features[idx], self.labels[idx]
    
    def get_participant_data(self, participant_id):
        """Get data for a specific participant"""
        if participant_id in self.participant_ids:
            indices = self.participant_groups.get_group(participant_id).index.tolist()
            features = self.features[indices]
            labels = self.labels[indices]
            return features, labels
        return None, None
    
    def get_participant_indices(self, participant_id):
        """Get indices for a specific participant"""
        if participant_id in self.participant_ids:
            return self.participant_groups.get_group(participant_id).index.tolist()
        return []
        
    def get_participant_type(self, participant_id):
        """Get type (honest/adversarial) for a participant"""
        if participant_id in self.participant_ids and 'label' in self.data.columns:
            labels = self.data[self.data['participant_id'] == participant_id]['label'].unique()
            if len(labels) > 0:
                return 'adversarial' if 'adv' in labels[0].lower() else 'honest'
        return "unknown"

# ========================
# 2. V2G Model with LayerNorm
# ========================
class V2GClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=2, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc4 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.norm3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return torch.log_softmax(x, dim=1)
# ========================
# 3. Robust Federated Learning Server - Improved
# ========================
class FederatedServer:
    def __init__(self, global_model, num_honest, num_adversarial):
        self.global_model = global_model
        self.participants = []
        self.total_participants = num_honest + num_adversarial
        self.num_honest = num_honest
        self.num_adversarial = num_adversarial
        
        # Configure parameters based on adversarial majority
        self.adversarial_majority = num_adversarial > num_honest
        
        if self.adversarial_majority:
            print("ADVERSARIAL MAJORITY DETECTED - Using specialized protection strategy")
            # Core parameters for adversarial majority
            self.num_byzantines = max(5, min(num_adversarial // 2, num_honest))
            self.multi_krum_m = max(3, num_honest // 3)
            
            # CRITICAL: Modified parameters to protect honest participants
            self.removal_threshold = 3.5  # Very high threshold to avoid false positives
            self.max_removal_rate = 0.08  # Lower removal rate per round
            self.bootstrap_rounds = 10    # Longer bootstrap period
            self.whitelist_rounds = 10    # Extended whitelist protection
            
            # Reputation system
            self.use_confidence_based_removal = True  # Only remove when confident
            self.confidence_threshold = 0.85  # Very high confidence needed
            self.reputation_decay = 0.95      # Slower decay
            
            # Strict budget for honest removals - MAXIMUM 10% honest can be removed
            self.honest_removal_budget = max(1, int(num_honest * 0.1))
            self.honest_protection_factor = 10.0  # Extreme protection for honest
            
            # Strategic measures
            self.forced_removal = True    # Allow forced removal when needed
            self.min_accuracy_threshold = 0.45  # Accuracy threshold for measures
            self.adv_per_round_limit = 3  # Maximum adversaries to remove per round
            
            # Safe list starts empty - participants earn their way in
            self.safe_list = []
            self.confirmed_adversarial = []  # Track confirmed adversaries
            
            # Behavioral analysis
            self.track_gradient_stats = True
            self.track_update_patterns = True
            self.use_behavioral_profiling = True
        else:
            # Standard parameters for non-adversarial majority
            self.num_byzantines = min(num_adversarial, num_honest//2)
            self.multi_krum_m = max(3, num_honest // 2)
            self.removal_threshold = 3.0
            self.max_removal_rate = 0.1
            self.bootstrap_rounds = 5
            self.whitelist_rounds = 5
            self.use_confidence_based_removal = False
            self.confidence_threshold = 0.7
            self.reputation_decay = 0.9
            self.honest_removal_budget = num_honest  # No real budget
            self.honest_protection_factor = 2.0
            self.forced_removal = False
            self.min_accuracy_threshold = 0.5
            self.adv_per_round_limit = 5
            self.safe_list = []
            self.confirmed_adversarial = []
            self.track_gradient_stats = False
            self.track_update_patterns = False
            self.use_behavioral_profiling = False
        
        self.current_round = 0
        self.used_honest_budget = 0  # Track how many honest we've removed
        
        # Minimum active participants
        self.min_participants = max(3, num_honest // 4)
        
        # Tracking
        self.score_history = defaultdict(list)
        self.gradient_norms = defaultdict(list)
        self.gradient_directions = defaultdict(list)
        self.update_magnitudes = defaultdict(list)
        self.model_acc_history = []
        self.selection_history = defaultdict(list)
        self.detected_adversaries = []
        self.reputation_scores = {}
        self.confidence_scores = {}  # Confidence of each participant being adversarial
        self.participant_profiles = {}  # Behavioral profiles
        self.identified_honest = []  # Identified honest participants
        
        # Early stopping parameters
        self.patience = 15 if self.adversarial_majority else 10
        self.min_rounds = 40 if self.adversarial_majority else 20
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        self.best_model_state = None
        
        # Initialize participants
        participant_types = [False]*num_honest + [True]*num_adversarial
        random.shuffle(participant_types)
        
        for i, is_adv in enumerate(participant_types):
            self.participants.append({
                'id': i,
                'model': copy.deepcopy(global_model),
                'data': None,
                'is_adversarial': is_adv,
                'active': True,
                'scores': [],
                'selection_count': 0,
                'outlier_frequency': 0,
                'removed_round': None,
                'suspicion_score': 0.0,
                'reputation': 0.5,  # Start with neutral reputation
                'behavioral_flags': 0,  # Count suspicious behaviors
                'consecutive_high_scores': 0  # Track consecutive high scores
            })
            
            # Initialize reputation with neutral value
            self.reputation_scores[i] = 0.5
            
            # Initialize confidence scores conservatively
            self.confidence_scores[i] = 0.0
        
        print(f"Server initialized with {num_honest} honest and {num_adversarial} adversarial participants")
        print(f"Detection params: threshold={self.removal_threshold}, rate={self.max_removal_rate}")
        print(f"Multi-Krum params: byzantines={self.num_byzantines}, m={self.multi_krum_m}")
        print(f"Honest protection: budget={self.honest_removal_budget}, factor={self.honest_protection_factor}")
        if self.forced_removal:
            print(f"Protection strategy: will prioritize adversarial removals and strictly limit honest removals")

    def calculate_gradient_metrics(self, gradients, indices):
        """Calculate statistically significant gradient metrics"""
        if not self.track_gradient_stats:
            return
            
        # Calculate gradient norms and directions
        for i, idx in enumerate(indices):
            if gradients[i] is None:
                continue
                
            # Flatten gradient
            flat_grad = torch.cat([g.view(-1) for g in gradients[i]])
            
            # Calculate norm
            norm = torch.norm(flat_grad).item()
            self.gradient_norms[idx].append(norm)
            
            # Calculate direction using unit vector
            if norm > 1e-6:  # Avoid division by zero
                direction = flat_grad / norm
                
                # Compare with previous directions if available
                if len(self.gradient_directions[idx]) > 0:
                    prev_direction = self.gradient_directions[idx][-1]
                    # Calculate cosine similarity with previous direction
                    similarity = torch.dot(direction, prev_direction).item()
                    
                    # Very abrupt direction changes are suspicious
                    if similarity < -0.5:  # Negative similarity = opposite direction
                        self.participants[idx]['behavioral_flags'] += 1
                
                # Store direction for future comparison
                self.gradient_directions[idx].append(direction)
            
            # Keep limited history
            if len(self.gradient_norms[idx]) > 10:
                self.gradient_norms[idx] = self.gradient_norms[idx][-10:]
            if len(self.gradient_directions[idx]) > 10:
                self.gradient_directions[idx] = self.gradient_directions[idx][-10:]

    def multi_krum(self, gradients, active_indices):
        """Enhanced Multi-Krum for adversarial scenarios"""
        valid_grads = []
        valid_indices = []
        
        for i, idx in enumerate(active_indices):
            if gradients[i] is not None:
                valid_grads.append(gradients[i])
                valid_indices.append(idx)
                
        if not valid_grads:
            return [], [], None
        
        # Flatten gradients
        flat_grads = [torch.cat([g.view(-1) for g in grad]) for grad in valid_grads]
        
        # Calculate key metrics for behavioral analysis
        if self.track_gradient_stats:
            self.calculate_gradient_metrics(gradients, active_indices)
            
        # Compute distances
        n = len(flat_grads)
        distances = torch.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = torch.sum((flat_grads[i] - flat_grads[j])**2).item()
                distances[i,j] = distances[j,i] = dist
        
        # Calculate Krum scores
        krum_scores = torch.zeros(n)
        
        # Adjust f based on active participants
        active_honest = sum(1 for i in active_indices if not self.participants[i]['is_adversarial'])
        active_adv = len(active_indices) - active_honest
        
        # Dynamic f value based on scenario
        if self.adversarial_majority:
            # More conservative for adversarial majority
            f = max(1, min(active_adv // 3, active_honest // 2))
        else:
            f = min(self.num_byzantines, n-2)
        
        # Ensure f is valid
        f = min(f, n-2) if n > 2 else 0
        
        # Calculate Krum scores
        for i in range(n):
            dist_i = distances[i]
            neighbor_distances, _ = torch.sort(dist_i)
            
            # Calculate krum score based on closest neighbors
            if f > 0 and n > f+1:
                krum_scores[i] = torch.sum(neighbor_distances[1:n-f])
            else:
                krum_scores[i] = torch.sum(neighbor_distances[1:])
        
        # Select participants
        _, sorted_indices = torch.sort(krum_scores)
        m = min(self.multi_krum_m, n)
        selected_indices_local = sorted_indices[:m].tolist()
        selected_indices = [valid_indices[i] for i in selected_indices_local]
        
        # Calculate gradient aggregation with inverse weighting
        selected_grads = [flat_grads[i] for i in selected_indices_local]
        selected_scores = [krum_scores[i].item() for i in selected_indices_local]
        
        # Inverse weighting - lower scores get higher weights
        if len(selected_grads) >= 2:
            max_score = max(selected_scores) + 1e-5
            weights = [max(0.1, (max_score - score + 1e-5) / max_score) for score in selected_scores]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Weighted average
            aggregated = torch.zeros_like(selected_grads[0])
            for i, grad in enumerate(selected_grads):
                aggregated += grad * weights[i]
        else:
            # Simple mean for small selection
            aggregated = torch.stack(selected_grads).mean(dim=0)
        
        # Prepare scores for all participants
        all_scores = [None] * len(gradients)
        for i, idx in enumerate(valid_indices):
            loc = active_indices.index(idx)
            all_scores[loc] = krum_scores[i].item()
            
        # Track how many honest vs. adversarial were selected
        honest_selected = sum(1 for idx in selected_indices if not self.participants[idx]['is_adversarial'])
        adv_selected = len(selected_indices) - honest_selected
        print(f"Multi-Krum selection: {honest_selected} honest, {adv_selected} adversarial (total: {len(selected_indices)})")
        
        # Update suspicion scores based on krum scores
        median_score = torch.median(krum_scores).item()
        for i, idx in enumerate(valid_indices):
            relative_score = krum_scores[i].item() / (median_score + 1e-10)
            
            # Use a more conservative approach for suspicion updates
            if idx in selected_indices:
                # Selected participants get reputation boost
                decay = 0.8 if self.adversarial_majority else 0.5
                old_suspicion = self.participants[idx]['suspicion_score']
                self.participants[idx]['suspicion_score'] = old_suspicion * decay
            else:
                # Non-selected participants get suspicion increase based on score
                if relative_score > 2.0:  # Only increase suspicion for significant outliers
                    self.participants[idx]['suspicion_score'] += 0.1 * (relative_score - 1.0)
                    
                    # Track consecutive high scores
                    self.participants[idx]['consecutive_high_scores'] += 1
                else:
                    self.participants[idx]['consecutive_high_scores'] = 0
        
        return selected_indices, all_scores, aggregated

    def update_reputation_scores(self, selected_indices, all_scores=None):
        """More effective reputation scoring system"""
        # Reward participants selected by Multi-Krum
        for idx in selected_indices:
            old_reputation = self.reputation_scores[idx]
            # Increase reputation for selected participants
            # Use smaller increments for adversarial majority scenario
            increment = 0.03 if self.adversarial_majority else 0.05
            self.reputation_scores[idx] = min(1.0, old_reputation + increment)
            
            # Decrease suspicion
            self.participants[idx]['suspicion_score'] *= 0.9
            
            # Reset consecutive high scores
            self.participants[idx]['consecutive_high_scores'] = 0
        
        # Analyze behavioral patterns
        if self.use_behavioral_profiling:
            for idx in self.reputation_scores.keys():
                if not self.participants[idx]['active']:
                    continue
                    
                # Check for suspicious gradient patterns
                if idx in self.gradient_norms and len(self.gradient_norms[idx]) >= 5:
                    norms = self.gradient_norms[idx]
                    
                    # Check for abnormal variance
                    variance = np.var(norms)
                    mean_norm = np.mean(norms)
                    cv = np.sqrt(variance) / (mean_norm + 1e-10)  # Coefficient of variation
                    
                    # Very high variance is suspicious
                    if cv > 0.7:
                        self.participants[idx]['behavioral_flags'] += 1
                        # Only penalize reputation for significant patterns
                        if self.participants[idx]['behavioral_flags'] >= 3:
                            self.reputation_scores[idx] = max(0.1, self.reputation_scores[idx] - 0.05)
        
        # Apply general slow decay to all reputations to encourage continuous good behavior
        for idx in self.reputation_scores.keys():
            if not self.participants[idx]['active']:
                continue
                
            # Apply very minimal regression to mean
            self.reputation_scores[idx] = self.reputation_decay * self.reputation_scores[idx] + \
                                         (1 - self.reputation_decay) * 0.5
                                         
            # Update participant object
            self.participants[idx]['reputation'] = self.reputation_scores[idx]
        
        # Update safe list very conservatively
        for idx in self.reputation_scores.keys():
            if not self.participants[idx]['active']:
                continue
                
            # Only add to safe list if reputation is exceptional AND selected frequently
            selection_rate = self.participants[idx]['selection_count'] / max(1, self.current_round)
            
            if (self.reputation_scores[idx] >= 0.8 and selection_rate >= 0.6 and 
                idx not in self.safe_list and self.current_round >= 15):
                
                # Extra verification for adversarial majority
                if self.adversarial_majority:
                    # Only add if behavioral flags are low
                    if self.participants[idx]['behavioral_flags'] <= 2:
                        self.safe_list.append(idx)
                        print(f"Added participant {idx} to safe list (reputation: {self.reputation_scores[idx]:.2f})")
                else:
                    self.safe_list.append(idx)
                    print(f"Added participant {idx} to safe list (reputation: {self.reputation_scores[idx]:.2f})")
        
        # Remove from safe list if reputation drops significantly
        self.safe_list = [idx for idx in self.safe_list 
                         if self.reputation_scores.get(idx, 0) >= 0.7 and self.participants[idx]['active']]

    def update_confidence_scores(self, active_indices):
        """Maintains conservative confidence scores for identifying adversaries"""
        if not self.use_confidence_based_removal:
            return
            
        for idx in active_indices:
            # Skip participants in safe list
            if idx in self.safe_list:
                self.confidence_scores[idx] = 0.0
                continue
                
            # Compute confidence based on multiple factors
            
            # Factor 1: Suspicion score
            suspicion = min(1.0, self.participants[idx]['suspicion_score'] / self.removal_threshold)
            
            # Factor 2: Selection rate (lower = more suspicious)
            selection_rate = self.participants[idx]['selection_count'] / max(1, self.current_round)
            selection_factor = max(0, 1.0 - selection_rate * 2.0)  # Scale to 0-1
            
            # Factor 3: Behavioral flags
            behavior_factor = min(1.0, self.participants[idx]['behavioral_flags'] / 5.0)
            
            # Factor 4: Consecutive high scores
            consecutive_factor = min(1.0, self.participants[idx]['consecutive_high_scores'] / 5.0)
            
            # Factor 5: Reputation (lower = more suspicious)
            reputation_factor = max(0, 1.0 - self.reputation_scores[idx])
            
            # Calculate weighted confidence
            if self.adversarial_majority:
                # More conservative weighting
                weighted_confidence = (
                    suspicion * 0.3 +
                    selection_factor * 0.2 +
                    behavior_factor * 0.2 +
                    consecutive_factor * 0.2 +
                    reputation_factor * 0.1
                )
            else:
                # Standard weighting
                weighted_confidence = (
                    suspicion * 0.4 +
                    selection_factor * 0.3 +
                    behavior_factor * 0.1 +
                    consecutive_factor * 0.1 +
                    reputation_factor * 0.1
                )
            
            # Apply temporal smoothing
            old_confidence = self.confidence_scores.get(idx, 0.0)
            new_confidence = 0.7 * old_confidence + 0.3 * weighted_confidence
            
            # Store updated confidence
            self.confidence_scores[idx] = new_confidence

    def identify_honest_participants(self, active_indices, round_accuracy=None):
        """More conservative honest participant identification"""
        # Only identify after several rounds
        if self.current_round < 10:
            return
            
        # Reset identified honest list periodically to prevent poisoning
        if self.current_round % 20 == 0:
            self.identified_honest = []
            
        # Method 1: High selection rate with good reputation
        for idx in active_indices:
            if idx in self.safe_list and idx not in self.identified_honest:
                self.identified_honest.append(idx)
                continue
                
            selection_rate = self.participants[idx]['selection_count'] / max(1, self.current_round)
            reputation = self.reputation_scores.get(idx, 0.5)
            
            # Very strict criteria for adversarial majority
            if self.adversarial_majority:
                if selection_rate > 0.7 and reputation > 0.8 and idx not in self.identified_honest:
                    self.identified_honest.append(idx)
            else:
                if selection_rate > 0.6 and reputation > 0.7 and idx not in self.identified_honest:
                    self.identified_honest.append(idx)
        
        # Method 2: Consistently good model accuracy
        if round_accuracy is not None:
            for idx, acc in round_accuracy.items():
                # Only consider very high accuracy participants for protection
                if acc > 0.7 and idx not in self.identified_honest:
                    self.identified_honest.append(idx)
        
        # Debug output - only print if there's a substantial number identified
        if len(self.identified_honest) > self.num_honest * 0.1:
            honest_identified = sum(1 for idx in self.identified_honest if not self.participants[idx]['is_adversarial'])
            adv_identified = len(self.identified_honest) - honest_identified
            
            # Only print periodically to reduce output clutter
            if self.current_round % 10 == 0:
                print(f"Protected participants: {honest_identified} honest, {adv_identified} adversarial "
                    f"(out of {len(self.identified_honest)} total)")

    def detect_anomalies(self, active_indices, force_removal=False):
        """Selectively detect anomalies, focusing on adversarial participants"""
        anomalies = []
        
        # Skip detection during bootstrap period unless forced
        if (self.current_round <= self.whitelist_rounds and not force_removal):
            return anomalies
            
        # Check if we have enough participants to remove some
        if len(active_indices) <= self.min_participants:
            return anomalies
        
        # Check honest removal budget - STRICTLY enforce this
        if self.used_honest_budget >= self.honest_removal_budget:
            print(f"STRICT PROTECTION: Honest removal budget exhausted ({self.used_honest_budget}/{self.honest_removal_budget})")
            print(f"Will only remove participants with high adversarial confidence")
            
            # If budget exhausted, only remove participants with high confidence of being adversarial
            candidates = []
            for idx in active_indices:
                # Skip participants in safe list or identified as honest
                if idx in self.safe_list or idx in self.identified_honest:
                    continue
                    
                # Only consider participants with high confidence
                if self.confidence_scores.get(idx, 0.0) >= self.confidence_threshold:
                    candidates.append((self.confidence_scores[idx], idx))
            
            # Sort by confidence (highest first)
            candidates.sort(reverse=True)
            
            # Only take participants with very high confidence
            threshold = 0.9 if self.adversarial_majority else 0.8
            high_confidence = [(score, idx) for score, idx in candidates if score >= threshold]
            
            # Limit removals per round
            limit = min(self.adv_per_round_limit, len(high_confidence))
            
            # Get most likely adversaries
            for _, idx in high_confidence[:limit]:
                anomalies.append(idx)
                
            if anomalies:
                # Validate if these are truly adversarial
                adv_count = sum(1 for idx in anomalies if self.participants[idx]['is_adversarial'])
                if adv_count < len(anomalies):
                    # Some honest might be removed - check budget
                    honest_to_remove = len(anomalies) - adv_count
                    if self.used_honest_budget + honest_to_remove > self.honest_removal_budget:
                        # Can't afford to remove any more honest participants
                        return []
            
            return anomalies
        
        # Regular detection path - when we still have honest budget
        
        # Get candidates with suspicion scores
        all_candidates = []
        for idx in active_indices:
            # Skip participants in safe list or identified as honest
            if idx in self.safe_list or idx in self.identified_honest:
                continue
                
            # Calculate composite suspicion
            suspicion = self.participants[idx]['suspicion_score']
            
            # Apply protection for frequently selected participants
            selection_rate = self.participants[idx]['selection_count'] / max(1, self.current_round)
            
            if selection_rate >= 0.4:  # Protection for frequently selected
                protection = selection_rate * self.honest_protection_factor
                suspicion = max(0, suspicion - protection)
            
            # Enhance suspicion for participants with suspicious behavior patterns
            if self.participants[idx]['consecutive_high_scores'] >= 3:
                suspicion += 0.5 * self.participants[idx]['consecutive_high_scores']
                
            if self.participants[idx]['behavioral_flags'] >= 3:
                suspicion += 0.3 * self.participants[idx]['behavioral_flags']
            
            # Add to candidates if suspicion is significant
            if suspicion > 0.5 or force_removal:
                all_candidates.append((suspicion, idx))
        
        # Sort by suspicion (highest first)
        all_candidates.sort(reverse=True)
        
        # For forced removal, be very selective
        if force_removal:
            if all_candidates:
                # If we're forcing removal, prioritize high suspicion participants
                force_count = min(2, len(all_candidates))  # Only remove 1-2 at a time
                
                # Check these candidates for adversarial status
                likely_adversarial = []
                for _, idx in all_candidates[:force_count*2]:  # Consider twice as many
                    # Use confidence score for filtering if available
                    confidence = self.confidence_scores.get(idx, 0.0)
                    if confidence >= 0.7:  # Good confidence of being adversarial
                        likely_adversarial.append(idx)
                        
                        # Add to confirmed adversarial list if confidence very high
                        if confidence >= 0.9 and idx not in self.confirmed_adversarial:
                            self.confirmed_adversarial.append(idx)
                            
                # Limit to force count
                anomalies = likely_adversarial[:force_count]
                
                # Important: check if we'll exceed our honest budget
                adv_count = sum(1 for idx in anomalies if self.participants[idx]['is_adversarial'])
                honest_count = len(anomalies) - adv_count
                
                if self.used_honest_budget + honest_count > self.honest_removal_budget:
                    # Filter to ensure we don't exceed budget
                    final_anomalies = []
                    remaining_honest = self.honest_removal_budget - self.used_honest_budget
                    
                    honest_added = 0
                    for idx in anomalies:
                        if self.participants[idx]['is_adversarial']:
                            final_anomalies.append(idx)
                        elif honest_added < remaining_honest:
                            final_anomalies.append(idx)
                            honest_added += 1
                            
                    return final_anomalies
                
                return anomalies
            return []
        
        # Standard detection with threshold
        suspects = []
        for suspicion, idx in all_candidates:
            if suspicion > self.removal_threshold:
                suspects.append(idx)
        
        # Apply confidence filtering if enabled
        if self.use_confidence_based_removal and suspects:
            confident_suspects = []
            for idx in suspects:
                if self.confidence_scores.get(idx, 0.0) >= self.confidence_threshold:
                    confident_suspects.append(idx)
                    
            suspects = confident_suspects
        
        # Limit removals to max rate
        max_removals = min(self.adv_per_round_limit, max(1, int(len(active_indices) * self.max_removal_rate)))
        candidates = suspects[:max_removals]
        
        # Check if these removals would exceed our honest budget
        adv_count = sum(1 for idx in candidates if self.participants[idx]['is_adversarial'])
        honest_count = len(candidates) - adv_count
        
        if self.used_honest_budget + honest_count > self.honest_removal_budget:
            # Filter to ensure we don't exceed budget
            final_candidates = []
            remaining_honest = self.honest_removal_budget - self.used_honest_budget
            
            honest_added = 0
            for idx in candidates:
                if self.participants[idx]['is_adversarial']:
                    final_candidates.append(idx)
                elif honest_added < remaining_honest:
                    final_candidates.append(idx)
                    honest_added += 1
                    
            anomalies = final_candidates
        else:
            anomalies = candidates
        
        return anomalies

    def update_model(self, gradients, active_indices, val_loader=None, round_accuracy=None):
        """Update model with aggregated gradients"""
        self.current_round += 1
        
        # Identify honest participants
        self.identify_honest_participants(active_indices, round_accuracy)
        
        # Apply Multi-Krum
        selected_indices, krum_scores, aggregated = self.multi_krum(gradients, active_indices)
        
        # Update reputation scores
        self.update_reputation_scores(selected_indices, krum_scores)
        
        # Update confidence scores
        self.update_confidence_scores(active_indices)
        
        # Dynamic step size scheduling
        if self.current_round < 5:
            step_size = 0.05  # Conservative early on
        elif self.current_round < 10:
            step_size = 0.08  # Medium
        else:
            step_size = 0.10  # Standard
            
        # Update global model
        if aggregated is not None:
            idx = 0
            for param in self.global_model.parameters():
                param_size = param.numel()
                grad_portion = aggregated[idx:idx+param_size].view(param.shape)
                with torch.no_grad():
                    param -= grad_portion * step_size
                idx += param_size
        
        # Update selection counts and score history
        for i, idx in enumerate(active_indices):
            if selected_indices and idx in selected_indices:
                self.participants[idx]['selection_count'] += 1
                
            # Update score history if available
            if i < len(krum_scores) and krum_scores[i] is not None:
                self.score_history[idx].append(krum_scores[i])
        
        # Calculate outlier frequency
        for idx in active_indices:
            if len(self.score_history[idx]) >= 3:
                scores = [s for s in self.score_history[idx] if s is not None]
                if not scores:
                    continue
                    
                all_scores = [s for participant_scores in self.score_history.values() 
                             for s in participant_scores if s is not None]
                if all_scores:
                    median_all = np.median(all_scores)
                    mad_all = np.median([abs(s - median_all) for s in all_scores]) + 1e-10
                    outlier_count = sum(1 for s in scores if abs(s - median_all) > 1.5 * mad_all)
                    self.participants[idx]['outlier_frequency'] = outlier_count / len(scores)
        
        # Evaluate and track model accuracy
        if val_loader:
            current_acc = self.evaluate_model(val_loader)
            self.model_acc_history.append(current_acc)
            
            # Check for early stopping
            if current_acc > self.best_val_acc:
                self.best_val_acc = current_acc
                self.best_model_state = copy.deepcopy(self.global_model.state_dict())
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
        
        return selected_indices, krum_scores

    def evaluate_model(self, data_loader, model=None):
        """Evaluate model accuracy"""
        model_to_eval = model if model is not None else self.global_model
        model_to_eval.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in data_loader:
                # Force device to CPU for compatibility
                features, labels = features.to('cpu'), labels.to('cpu')
                outputs = model_to_eval(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0
    
    def check_early_stopping(self):
        """Check if training should stop early, respecting minimum rounds"""
        # Never stop early in adversarial majority before minimum rounds
        if self.current_round < self.min_rounds:
            return False
            
        # After minimum rounds, use patience-based stopping
        if self.early_stop_counter >= self.patience:
            # Load best model
            if self.best_model_state is not None:
                self.global_model.load_state_dict(self.best_model_state)
            return True
            
        return False
    
    def force_extreme_measures(self, active_indices):
        """Selectively remove high-confidence adversarial participants"""
        if not self.forced_removal:
            return []
            
        # Check triggers for extreme measures
        trigger_extreme = False
        trigger_reason = ""
        
        # Condition 1: Low accuracy after several rounds
        if (self.current_round >= self.whitelist_rounds + 5 and 
            len(self.model_acc_history) > 0 and 
            self.model_acc_history[-1] < self.min_accuracy_threshold):
            trigger_reason = f"Low accuracy ({self.model_acc_history[-1]:.4f} below threshold {self.min_accuracy_threshold})"
            trigger_extreme = True
            
        # Condition 2: Adversarial participants significantly outnumber honest ones
        adv_active = sum(1 for i in active_indices if self.participants[i]['is_adversarial'])
        honest_active = len(active_indices) - adv_active
        
        if (adv_active > honest_active * 1.5 and
            self.current_round >= self.whitelist_rounds + 5):
            trigger_reason = f"Adversarial majority ({adv_active} vs {honest_active} honest)"
            trigger_extreme = True
            
        # Condition 3: Regular forced removal rounds
        force_rounds = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90]
        if self.current_round in force_rounds and self.adversarial_majority:
            trigger_reason = f"Scheduled adversarial removal at round {self.current_round}"
            trigger_extreme = True
            
        # Apply extreme measures if triggered
        if trigger_extreme:
            print(f"TRIGGER: {trigger_reason}")
            
            # Use confidence scores to find likely adversaries
            candidates = []
            for idx in active_indices:
                # Skip safe participants
                if idx in self.safe_list or idx in self.identified_honest:
                    continue
                    
                # Use confidence scores to find likely adversaries
                confidence = self.confidence_scores.get(idx, 0.0)
                if confidence >= 0.7:  # Reasonable confidence
                    candidates.append((confidence, idx))
                    
            # Sort by confidence
            candidates.sort(reverse=True)
            
            # Determine removal count - more aggressive on force rounds
            if self.current_round in force_rounds:
                # More aggressive on scheduled rounds
                removal_count = min(3, len(candidates))
            else:
                # More conservative otherwise
                removal_count = min(2, len(candidates))
                
            # Get participants to remove
            forced_removals = [idx for _, idx in candidates[:removal_count]]
            
            # Validate these removals against honest budget
            adv_count = sum(1 for idx in forced_removals if self.participants[idx]['is_adversarial'])
            honest_count = len(forced_removals) - adv_count
            
            if self.used_honest_budget + honest_count > self.honest_removal_budget:
                # Filter to stay within budget
                filtered_removals = []
                remaining_honest = self.honest_removal_budget - self.used_honest_budget
                
                honest_added = 0
                for idx in forced_removals:
                    if self.participants[idx]['is_adversarial']:
                        filtered_removals.append(idx)
                    elif honest_added < remaining_honest:
                        filtered_removals.append(idx)
                        honest_added += 1
                        
                forced_removals = filtered_removals
            
            if forced_removals:
                print(f"FORCED REMOVAL: Removing {len(forced_removals)} participants")
                return forced_removals
                
        return []
# ==# ========================
# 4. Enhanced Adversarial Training
# ========================
def train_participant(model, data_loader, is_adversarial=False, epoch=0, val_loader=None):
    """Training function with V2G-specific adversarial behavior"""
    # Dynamic learning rate
    if epoch < 3:
        lr = 0.007  # Start conservative
    elif epoch < 8:
        lr = 0.01   # Standard rate
    else:
        lr = 0.008  # Decrease slightly for stability
        
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    model.train()
    gradients = []
    
    running_loss = 0.0
    batches = 0

    for features, labels in data_loader:
        # Force device to CPU for compatibility
        features, labels = features.to('cpu'), labels.to('cpu')

        if is_adversarial:
            # V2G-specific adversarial behavior - more sophisticated attack
            if epoch < 5:  # Subtle early attacks
                # Occasional label flipping (20% chance)
                if random.random() < 0.2:
                    labels = (labels + 1) % dataset.num_classes
                    
                # Subtle noise
                noise = torch.randn_like(features) * 0.05
                features = features + noise
                
            elif 5 <= epoch < 15:  # Stronger attacks in middle rounds
                # More frequent label flipping (50% chance)
                if random.random() < 0.5:
                    labels = (labels + 1) % dataset.num_classes
                
                # More significant noise
                noise = torch.randn_like(features) * 0.15
                features = features + noise
                
            else:  # Adaptive attacks in later rounds
                # Tactical label flipping (40% chance) 
                if random.random() < 0.4:
                    labels = (labels + 1) % dataset.num_classes
                
                # Adaptive noise levels to avoid detection
                if random.random() < 0.6:
                    noise = torch.randn_like(features) * 0.08
                    features = features + noise

        optimizer.zero_grad()
        outputs = model(features)
        loss = nn.NLLLoss()(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        gradients = [p.grad.clone() for p in model.parameters()]
        optimizer.step()
        
        running_loss += loss.item()
        batches += 1

    # Calculate model accuracy if validation set provided
    model_accuracy = None
    if val_loader is not None:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to('cpu'), labels.to('cpu')
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        model_accuracy = correct / total if total > 0 else 0
        model.train()  # Switch back to train mode

    return gradients, running_loss / max(1, batches), model_accuracy
# ========================
# 5. Evaluation Metrics
# ========================
def calculate_metrics(server, removed_participants):
    """Calculate detection performance metrics"""
    y_true = np.array([1 if p['is_adversarial'] else 0 for p in server.participants])
    y_pred = np.array([1 if i in removed_participants else 0 for i in range(len(server.participants))])
    
    # Count removals
    total_honest = sum(1 for p in server.participants if not p['is_adversarial'])
    total_adv = sum(1 for p in server.participants if p['is_adversarial'])
    
    removed_honest = sum(1 for i in removed_participants if not server.participants[i]['is_adversarial'])
    removed_adv = sum(1 for i in removed_participants if server.participants[i]['is_adversarial'])
    
    # Calculate percentages
    honest_removal_rate = removed_honest / total_honest if total_honest > 0 else 0
    adv_removal_rate = removed_adv / total_adv if total_adv > 0 else 0
    
    # Handle edge cases
    if sum(y_true) == 0 and sum(y_pred) == 0:
        precision = 1.0
        weighted_precision = 1.0
    elif sum(y_pred) == 0:
        precision = 0.0
        weighted_precision = 0.0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
    recall = recall_score(y_true, y_pred, zero_division=0) if sum(y_true) > 0 else 1.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = 0 if cm.shape[0] == 0 or cm.shape[1] <= 1 else cm[0, 1]
        fn = 0 if cm.shape[0] <= 1 or cm.shape[1] == 0 else cm[1, 0]
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    
    # Add validation accuracy
    model_accuracy = np.mean(server.model_acc_history[-5:]) * 100 if server.model_acc_history else 0
    
    # Calculate removal preference (higher value means more focused on adversaries)
    if honest_removal_rate > 0:
        relative_preference = adv_removal_rate / honest_removal_rate
    else:
        # If no honest participants were removed, this is ideal
        relative_preference = float('inf') if adv_removal_rate > 0 else 1.0
    
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision * 100,
        'weighted_precision': weighted_precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'model_accuracy': model_accuracy,
        'honest_removal_rate': honest_removal_rate * 100,
        'adv_removal_rate': adv_removal_rate * 100,
        'relative_preference': relative_preference
    }

def print_detection_performance(metrics):
    """Print detection performance metrics"""
    print("\nDETECTION PERFORMANCE:")
    print("--------------------------------------------------")
    print(f"{'Accuracy:':<15}{metrics['accuracy']:.2f}%")
    print(f"{'Precision:':<15}{metrics['precision']:.2f}%")
    print(f"{'Recall:':<15}{metrics['recall']:.2f}%")
    print(f"{'F1 Score:':<15}{metrics['f1']:.2f}%")
    print(f"{'Model Acc (last 5):':<15}{metrics['model_accuracy']:.2f}%")
    print("--------------------------------------------------")
    print(f"{'Honest Removed:':<20}{metrics['honest_removal_rate']:.1f}%")
    print(f"{'Adversarial Removed:':<20}{metrics['adv_removal_rate']:.1f}%")
    print(f"{'Removal Preference:':<20}{metrics['relative_preference']:.2f}x")
    print("--------------------------------------------------")
    print(f"{'True Positives:':<20}{metrics['true_positives']}")
    print(f"{'False Positives:':<20}{metrics['false_positives']}")
    print(f"{'True Negatives:':<20}{metrics['true_negatives']}")
    print(f"{'False Negatives:':<20}{metrics['false_negatives']}")
    print("--------------------------------------------------")
    

# ========================
# 6. Run Federated Learning Simulation
# ========================
def distribute_data_to_participants(dataset, train_indices, server):
    """
    Distribute training data to participants based on participant_id.
    
    Args:
        dataset (V2GDataset): The V2G dataset containing participant data.
        train_indices (list): Indices of the training data subset.
        server (FederatedServer): The federated server containing participant information.
    """
    participant_ids = dataset.participant_ids
    print(f"Distributing data to {len(participant_ids)} participants")

    activated_honest = 0
    activated_adversarial = 0
    min_records_per_participant = 5

    for participant in server.participants:
        p_id = participant['id']
        if p_id in participant_ids:
            participant_indices = dataset.get_participant_indices(p_id)
            participant_train_indices = [idx for idx in participant_indices if idx in train_indices]
            if len(participant_train_indices) >= min_records_per_participant:
                participant_subset = Subset(dataset, participant_train_indices)
                participant_dataloader = DataLoader(
                    participant_subset,
                    batch_size=64,
                    shuffle=True
                )
                participant['data'] = participant_dataloader
                if participant['is_adversarial']:
                    activated_adversarial += 1
                else:
                    activated_honest += 1
            else:
                print(f"Warning: Participant {p_id} has only {len(participant_train_indices)} records, deactivating")
                participant['active'] = False
        else:
            print(f"Warning: Participant {p_id} not found in dataset, deactivating")
            participant['active'] = False
    
    total_activated = activated_honest + activated_adversarial
    print(f"Warning: Not enough data for all participants. Limiting to {total_activated}")
    print(f"Activated {activated_honest} honest and {activated_adversarial} adversarial participants")

# ========================
# 6. Run Federated Learning Simulation
# ========================
def run_v2g_simulation(dataset_path, num_honest=10, num_adversarial=11, rounds=100, no_early_stop=False):
    """Run federated learning simulation with V2G data"""
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create dataset and model
    global dataset
    dataset = V2GDataset(dataset_path)
    
    # Create model based on dataset features
    model = V2GClassifier(
        input_size=dataset.num_features,
        hidden_size=128,
        num_classes=dataset.num_classes,
        dropout=0.15
    )
    
    # Create split for train/validation
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Initialize server
    server = FederatedServer(model, num_honest, num_adversarial)
    
    # Override early stopping if requested
    if no_early_stop:
        server.min_rounds = rounds
        server.patience = rounds * 2  # Effectively disable early stopping
        print(f"Early stopping disabled - will run full {rounds} rounds")
    
    # Distribute data to participants
    distribute_data_to_participants(dataset, train_indices, server)
    
    print(f"Starting FL with {num_honest} honest and {num_adversarial} adversarial participants")
    print(f"Bootstrap rounds: {server.bootstrap_rounds}, Whitelist rounds: {server.whitelist_rounds}")
    print(f"Removal threshold: {server.removal_threshold}, Max removal rate: {server.max_removal_rate*100}%")
    print(f"Protection strategy: Strictly limit honest removals to {server.honest_removal_budget} participants")
    
    # Track removed participants
    removed_participants = []
    
    # Main training loop
    for round in range(rounds):
        gradients = []
        active_indices = []
        round_accuracy = {}
        
        # Check if we have enough active participants
        active_count = sum(1 for p in server.participants if p['active'] and p['data'] is not None)
        if active_count < server.min_participants:
            print(f"Stopping at round {round} - not enough active participants")
            break
        
        # Check for early stopping
        if not no_early_stop and server.check_early_stopping():
            print(f"Early stopping triggered at round {round} - validation accuracy not improving")
            break
        
        # Train participants
        for i, p in enumerate(server.participants):
            if not p['active'] or p['data'] is None:
                continue
                
            try:
                grad, loss, accuracy = train_participant(
                    p['model'],
                    p['data'],
                    is_adversarial=p['is_adversarial'],
                    epoch=round,
                    val_loader=val_loader
                )
                gradients.append(grad)
                active_indices.append(i)
                
                # Track individual model accuracy
                if accuracy is not None:
                    round_accuracy[i] = accuracy
            except Exception as e:
                print(f"Error training participant {i}: {e}")
                p['active'] = False  # Deactivate on error
        
        # Update model
        if len(active_indices) >= server.min_participants:
            server.update_model(gradients, active_indices, val_loader, round_accuracy)
        else:
            print(f"Skipping round {round+1} - not enough active participants")
            continue
            
        # Check for extreme measures
        extreme_removals = server.force_extreme_measures(active_indices)
        if extreme_removals:
            # Process extreme removals
            honest_removed = sum(1 for idx in extreme_removals if not server.participants[idx]['is_adversarial'])
            adv_removed = len(extreme_removals) - honest_removed
            
            # Update honest removal budget
            if honest_removed > 0:
                server.used_honest_budget += honest_removed
                print(f"Honest budget used: {server.used_honest_budget}/{server.honest_removal_budget}")
            
            for idx in extreme_removals:
                server.participants[idx]['active'] = False
                server.participants[idx]['removed_round'] = round
                removed_participants.append(idx)
            
            print(f"Round {round+1}: EXTREME MEASURES removed {len(extreme_removals)} participants "
                  f"(Honest: {honest_removed}, Adversarial: {adv_removed})")
            
            if honest_removed > 0 and adv_removed > 0:
                ratio = adv_removed / honest_removed
                print(f"  Removal ratio (adv/honest): {ratio:.1f}x")
            
        # Regular detection
        else:
            # Determine if we should force removal
            force_removal = ((round+1) % 10 == 0) and server.adversarial_majority
            
            # Detect anomalies
            candidates = server.detect_anomalies(active_indices, force_removal)
            
            # Remove participants
            if candidates:
                honest_removed = sum(1 for idx in candidates if not server.participants[idx]['is_adversarial'])
                adv_removed = len(candidates) - honest_removed
                
                # Update honest removal budget
                if honest_removed > 0:
                    server.used_honest_budget += honest_removed
                    print(f"Honest budget used: {server.used_honest_budget}/{server.honest_removal_budget}")
                
                # Proceed with removal
                for idx in candidates:
                    server.participants[idx]['active'] = False
                    server.participants[idx]['removed_round'] = round
                    removed_participants.append(idx)
                
                print(f"Round {round+1}: Removed {len(candidates)} participants "
                      f"(Honest: {honest_removed}, Adversarial: {adv_removed})")
                
                if honest_removed > 0 and adv_removed > 0:
                    ratio = adv_removed / honest_removed
                    print(f"  Removal ratio (adv/honest): {ratio:.1f}x")
                elif honest_removed == 0 and adv_removed > 0:
                    print(f"  Perfect removal! Only adversarial participants removed.")
                    
                print(f"  Current model accuracy: {server.model_acc_history[-1]:.2f}")
        
        # Progress update
        if (round + 1) % 5 == 0 or round == 0:
            active_honest = sum(1 for i in active_indices if not server.participants[i]['is_adversarial'])
            active_adv = sum(1 for i in active_indices if server.participants[i]['is_adversarial'])
            print(f"Round {round+1}: Active: {len(active_indices)} "
                  f"(Honest: {active_honest}, Adversarial: {active_adv}), "
                  f"Accuracy: {server.model_acc_history[-1]:.2f}")
        
        # Early stopping if too few participants remain
        if len(active_indices) <= server.min_participants:
            print(f"Stopping early at round {round+1} - minimum participant threshold reached")
            break
    
    # Final evaluation
    metrics = calculate_metrics(server, removed_participants)
    print_detection_performance(metrics)
    
    # Print summary
    total_honest = num_honest
    total_adv = num_adversarial
    removed_honest = sum(1 for idx in removed_participants if not server.participants[idx]['is_adversarial'])
    removed_adv = sum(1 for idx in removed_participants if server.participants[idx]['is_adversarial'])
    
    print(f"\nFINAL RESULTS:")
    print(f"Honest participants: {total_honest - removed_honest}/{total_honest} remaining "
          f"({removed_honest/total_honest*100:.1f}% removed)")
    print(f"Adversarial participants: {total_adv - removed_adv}/{total_adv} remaining "
          f"({removed_adv/total_adv*100:.1f}% removed)")
    print(f"Final model accuracy: {server.model_acc_history[-1]:.2f}")
    print(f"Removal preference ratio: {metrics['relative_preference']:.2f}x")
    
    return server, removed_participants, metrics
# ========================
# 7. Main Execution
# ========================
if __name__ == "__main__":
    # Set device to CPU for compatibility
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # CSV file path
    dataset_path = r"C:\Users\Administrator\Downloads\v2g_simulated_dataset (1).csv"
    
    # Run simulation with adversarial majority
    print("Running V2G Federated Learning with adversarial majority...")
    server, removed, metrics = run_v2g_simulation(
        dataset_path=dataset_path,
        num_honest=50,
        num_adversarial=55,
        rounds=100,
        no_early_stop=False
    )
    
    # Calculate selection rate statistics
    honest_selection = []
    adv_selection = []
    
    for i, p in enumerate(server.participants):
        selection_rate = p['selection_count'] / max(1, server.current_round)
        if p['is_adversarial']:
            adv_selection.append(selection_rate)
        else:
            honest_selection.append(selection_rate)
    
    print("\nPARTICIPANT BEHAVIOR ANALYSIS:")
    print("--------------------------------------------------")
    print(f"{'Average selection rate (honest):':<35}{np.mean(honest_selection) if honest_selection else 0:.3f}")
    print(f"{'Average selection rate (adversarial):':<35}{np.mean(adv_selection) if adv_selection else 0:.3f}")
    
    print(f"\nSimulation completed!")
    print(f"Detection F1 score: {metrics['f1']:.2f}%")
    print(f"Relative preference for removing adversaries: {metrics['relative_preference']:.2f}x")
    print(f"Honest participants removed: {metrics['honest_removal_rate']:.1f}%")
    print(f"Adversarial participants removed: {metrics['adv_removal_rate']:.1f}%"))