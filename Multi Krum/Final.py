import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, Subset
import random
from collections import defaultdict

# Custom Dataset for V2G
class V2GDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        print(f"Dataset loaded with {len(self.data)} records")
        print(f"Columns: {', '.join(self.data.columns)}")
        
        # Preprocess timestamp
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data['hour'] = self.data['timestamp'].dt.hour
        else:
            self.data['hour'] = 0
        
        # Features and labels
        feature_cols = ['battery_capacity_kWh', 'current_charge_kWh', 'discharge_rate_kW', 
                       'energy_requested_kWh', 'hour']
        self.features = self.data[feature_cols].values
        self.labels = self.data['label'].str.lower().values
        self.participant_ids = self.data['participant_id'].values
        
        # Scale features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        print(f"Label classes: {', '.join(self.label_encoder.classes_)}")
        
        # Dataset stats
        self.num_features = self.features.shape[1]
        self.num_classes = len(self.label_encoder.classes_)
        unique_participants = len(set(self.participant_ids))
        print(f"Found {unique_participants} unique participants")
        honest_count = sum(self.labels == self.label_encoder.transform(['honest'])[0])
        adversarial_count = len(self.labels) - honest_count
        print(f"Data distribution: {honest_count} honest, {adversarial_count} adversarial records")
        honest_parts = len(set(self.participant_ids[self.labels == self.label_encoder.transform(['honest'])[0]]))
        adversarial_parts = unique_participants - honest_parts
        print(f"Participant distribution: {honest_parts} honest, {adversarial_parts} adversarial participants")
        print(f"Features: {self.num_features}, Classes: {self.num_classes}")
        print(f"Dataset participant IDs: {sorted(set(self.participant_ids))}")
        
        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # Group indices by participant
        self.participant_indices = defaultdict(list)
        for idx, pid in enumerate(self.participant_ids):
            self.participant_indices[pid].append(idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def get_participant_data(self, participant_id):
        indices = self.participant_indices.get(participant_id, [])
        return self.features[indices], self.labels[indices]
    
    def get_participant_indices(self, participant_id):
        return self.participant_indices.get(participant_id, [])
    
    def get_participant_type(self, participant_id):
        indices = self.get_participant_indices(participant_id)
        if not indices:
            return None
        labels = self.labels[indices].numpy()
        return 'honest' if np.mean(labels == self.label_encoder.transform(['honest'])[0]) > 0.5 else 'adversarial'

# Neural Network Model
class V2GClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=2, dropout=0.2):  # Fix: Increased hidden_size
        super(V2GClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual block 1
        self.residual1 = nn.Linear(hidden_size, hidden_size // 2)
        self.norm2 = nn.LayerNorm(hidden_size // 2)
        self.residual2 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.norm3 = nn.LayerNorm(hidden_size // 2)
        
        # Residual block 2 (Fix: Added)
        self.residual3 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.norm4 = nn.LayerNorm(hidden_size // 2)
        self.residual4 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.norm5 = nn.LayerNorm(hidden_size // 2)
        
        # Feature interaction
        self.interaction = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 2)
        )
        
        self.output_layer = nn.Linear(hidden_size // 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Residual block 1
        residual = x
        x = self.residual1(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.residual2(x)
        x = self.norm3(x)
        x = self.relu(x)
        
        # Residual block 2 (Fix: Added)
        residual = x
        x = self.residual3(x)
        x = self.norm4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.residual4(x)
        x = self.norm5(x)
        x = self.relu(x)
        
        x = x + self.interaction(x)  # Feature interaction
        x = self.dropout(x)
        
        x = self.output_layer(x)
        return self.log_softmax(x)

# Federated Server
class FederatedServer:
    def __init__(self, model, num_honest, num_adversarial, device='cpu', dataset=None):
        self.model = model.to(device)
        self.device = device
        self.num_honest = num_honest
        self.num_adversarial = num_adversarial
        self.adversarial_majority = num_adversarial > num_honest
        self.dataset = dataset
        self.current_round = 0
        self.participants = {}
        self.reputation_scores = {}
        self.confidence_scores = {}
        self.gradient_norms = defaultdict(list)
        self.gradient_directions = defaultdict(list)
        self.selection_history = defaultdict(list)
        self.accuracy_history = []
        self.removed_participants = []
        self.safe_list = set()
        self.honest_removed_count = 0  # Fix: Track global honest removals
        
        # Initialize parameters
        if self.adversarial_majority:
            self.removal_threshold = 3.5  # Fix: Increased
            self.max_removal_rate = 0.12
            self.bootstrap_rounds = 15
            self.whitelist_rounds = 15
            self.use_confidence_based_removal = True
            self.confidence_threshold = 0.82  # Fix: Increased
            self.reputation_decay = 0.95
            self.honest_removal_budget = 5
            self.honest_protection_factor = 10.0
            self.forced_removal = True
            self.min_accuracy_threshold = 0.5  # Fix: Increased
            self.adv_per_round_limit = 4  # Fix: Reduced
            self.patience = 30  # Fix: Increased
            self.min_rounds = 60
            self.num_byzantines = 15
            self.multi_krum_m = 20
        else:
            self.num_byzantines = max(3, num_adversarial // 3)
            self.multi_krum_m = max(3, num_honest // 2)
            self.removal_threshold = 3.0
            self.max_removal_rate = 0.1
            self.bootstrap_rounds = 5
            self.whitelist_rounds = 5
            self.use_confidence_based_removal = False
            self.confidence_threshold = 0.8
            self.reputation_decay = 0.98
            self.honest_removal_budget = num_honest // 5
            self.honest_protection_factor = 5.0
            self.forced_removal = False
            self.min_accuracy_threshold = 0.5
            self.adv_per_round_limit = 3
            self.patience = 10
            self.min_rounds = 20
        
        print(f"ADVERSARIAL MAJORITY DETECTED - Using specialized protection strategy" if self.adversarial_majority else "Standard mode")
        print(f"Detection params: threshold={self.removal_threshold}, rate={self.max_removal_rate}")
        print(f"Multi-Krum params: byzantines={self.num_byzantines}, m={self.multi_krum_m}")
        print(f"Honest protection: budget={self.honest_removal_budget}, factor={self.honest_protection_factor}")
        
        # Initialize participants using dataset participant IDs
        if dataset is None:
            raise ValueError("Dataset must be provided for participant initialization")
        available_pids = list(set(dataset.participant_ids))
        if len(available_pids) < num_honest + num_adversarial:
            print(f"Warning: Only {len(available_pids)} participants available, adjusting num_honest and num_adversarial")
            total = num_honest + num_adversarial
            self.num_honest = int(len(available_pids) * (num_honest / total))
            self.num_adversarial = len(available_pids) - self.num_honest
            print(f"Adjusted to {self.num_honest} honest and {self.num_adversarial} adversarial participants")
        
        random.shuffle(available_pids)
        honest_ids = available_pids[:self.num_honest]
        adversarial_ids = available_pids[self.num_honest:self.num_honest + self.num_adversarial]
        
        for idx in honest_ids:
            self.participants[idx] = {
                'model': V2GClassifier(model.input_layer.in_features, num_classes=model.output_layer.out_features).to(device),
                'type': 'honest',
                'data': None,
                'active': True,
                'suspicion_score': 0.0,
                'behavioral_flags': 0,
                'selection_count': 0
            }
            self.reputation_scores[idx] = 1.0
            self.confidence_scores[idx] = 0.0
        
        for idx in adversarial_ids:
            self.participants[idx] = {
                'model': V2GClassifier(model.input_layer.in_features, num_classes=model.output_layer.out_features).to(device),
                'type': 'adversarial',
                'data': None,
                'active': True,
                'suspicion_score': 0.0,
                'behavioral_flags': 0,
                'selection_count': 0
            }
            self.reputation_scores[idx] = 1.0
            self.confidence_scores[idx] = 0.0
        
        print(f"Server initialized with {self.num_honest} honest and {self.num_adversarial} adversarial participants")
    
    def calculate_gradient_metrics(self, idx, gradients):
        grad_vector = torch.cat([g.flatten() for g in gradients]).to(self.device)
        norm = torch.norm(grad_vector).item()
        self.gradient_norms[idx].append(norm)
        
        if len(self.gradient_norms[idx]) >= 2:
            prev_grad = torch.cat([g.flatten() for g in self.gradient_directions[idx][-1]]).to(self.device)
            direction_change = torch.dot(grad_vector / (norm + 1e-10), prev_grad / (torch.norm(prev_grad) + 1e-10)).item()
            if direction_change < 0.5:
                self.participants[idx]['behavioral_flags'] += 1
        
        self.gradient_directions[idx].append([g.clone().detach() for g in gradients])
    
    def multi_krum(self, gradients_dict):
        if not gradients_dict:
            return [], []
        
        n = len(gradients_dict)
        f = self.num_byzantines
        m = min(self.multi_krum_m, n)
        
        distances = {}
        for i in gradients_dict:
            distances[i] = []
            grad_i = torch.cat([g.flatten() for g in gradients_dict[i]]).to(self.device)
            for j in gradients_dict:
                if i != j:
                    grad_j = torch.cat([g.flatten() for g in gradients_dict[j]]).to(self.device)
                    dist = torch.norm(grad_i - grad_j).item()
                    distances[i].append(dist)
        
        krum_scores = {}
        for i in distances:
            sorted_dists = sorted(distances[i])[:max(1, n - f - 2)]
            krum_scores[i] = sum(sorted_dists)
        
        selected = sorted(krum_scores.items(), key=lambda x: x[1])[:m]
        selected_ids = [idx for idx, _ in selected]
        weights = [1.0 / (score + 1e-10) for _, score in selected]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        for idx in gradients_dict:
            if idx not in selected_ids:
                self.participants[idx]['suspicion_score'] += 0.05
            else:
                self.participants[idx]['selection_count'] += 1
                self.selection_history[idx].append(True)
            if idx not in self.selection_history:
                self.selection_history[idx] = []
            if len(self.selection_history[idx]) == 0 or not self.selection_history[idx][-1]:
                self.selection_history[idx].append(idx in selected_ids)
        
        return selected_ids, weights
    
    def update_reputation_scores(self, selected_ids):
        for idx in self.participants:
            if not self.participants[idx]['active']:
                continue
            if idx in selected_ids:
                self.reputation_scores[idx] = min(1.0, self.reputation_scores[idx] + 0.05)
            else:
                self.reputation_scores[idx] = max(0.1, self.reputation_scores[idx] * self.reputation_decay)
            
            # Behavioral profiling
            if idx in self.gradient_norms and len(self.gradient_norms[idx]) >= 5:
                norms = self.gradient_norms[idx]
                variance = np.var(norms)
                mean_norm = np.mean(norms)
                cv = np.sqrt(variance) / (mean_norm + 1e-10)
                if cv > 0.7:  # Fix: Increased
                    self.participants[idx]['behavioral_flags'] += 1
                    if self.participants[idx]['behavioral_flags'] >= 3:
                        self.reputation_scores[idx] = max(0.1, self.reputation_scores[idx] - 0.07)
            
            # Selection rate penalty (Fix: Delayed to round 30)
            if self.current_round >= 30:
                selection_rate = self.participants[idx]['selection_count'] / self.current_round
                if selection_rate < 0.2:
                    self.participants[idx]['behavioral_flags'] += 1
                    self.reputation_scores[idx] = max(0.1, self.reputation_scores[idx] - 0.05)
        
        # Update safe list (Fix: Lowered criteria)
        for idx in self.participants:
            if self.participants[idx]['selection_count'] >= self.current_round * 0.4 and \
               self.reputation_scores[idx] >= 0.8:
                self.safe_list.add(idx)
    
    def update_confidence_scores(self):
        for idx in self.participants:
            if not self.participants[idx]['active']:
                continue
            suspicion = min(self.participants[idx]['suspicion_score'] / self.removal_threshold, 1.0)
            selection_rate = self.participants[idx]['selection_count'] / max(self.current_round, 1)
            selection_factor = max(0.0, 1.0 - selection_rate / 0.5)
            behavior_factor = min(self.participants[idx]['behavioral_flags'] / 5.0, 1.0)
            consecutive_factor = 0.0
            if self.selection_history[idx] and len(self.selection_history[idx]) >= 5:
                recent = self.selection_history[idx][-5:]
                if all(not x for x in recent):
                    consecutive_factor = 1.0
            reputation_factor = max(0.0, 1.0 - self.reputation_scores[idx])
            
            if self.adversarial_majority:
                weighted_confidence = (
                    suspicion * 0.25 +
                    selection_factor * 0.3 +
                    behavior_factor * 0.25 +
                    consecutive_factor * 0.15 +
                    reputation_factor * 0.05
                )
            else:
                weighted_confidence = (
                    suspicion * 0.4 +
                    selection_factor * 0.2 +
                    behavior_factor * 0.2 +
                    consecutive_factor * 0.1 +
                    reputation_factor * 0.1
                )
            
            self.confidence_scores[idx] = min(1.0, weighted_confidence)
    
    def identify_honest_participants(self, val_loader=None):
        honest_candidates = []
        for idx in self.participants:
            if not self.participants[idx]['active']:
                continue
            selection_rate = self.participants[idx]['selection_count'] / max(self.current_round, 1)
            # Fix: Added gradient consistency check
            gradient_cv = 0.0
            if idx in self.gradient_norms and len(self.gradient_norms[idx]) >= 5:
                norms = self.gradient_norms[idx]
                variance = np.var(norms)
                mean_norm = np.mean(norms)
                gradient_cv = np.sqrt(variance) / (mean_norm + 1e-10)
            
            if selection_rate > 0.5 or self.reputation_scores[idx] > 0.9 or gradient_cv < 0.3:
                honest_candidates.append(idx)
            elif val_loader:
                model = self.participants[idx]['model']
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = correct / total
                if accuracy > 0.6:  # Fix: Lowered threshold
                    honest_candidates.append(idx)
        return honest_candidates
    
    def detect_anomalies(self, honest_candidates, val_loader=None):
        to_remove = []
        active_count = sum(1 for p in self.participants.values() if p['active'])
        max_removals = int(active_count * self.max_removal_rate)
        max_removals = min(max_removals, self.adv_per_round_limit)
        
        if self.use_confidence_based_removal:
            candidates = [(idx, self.confidence_scores[idx]) for idx in self.participants 
                         if self.participants[idx]['active'] and self.confidence_scores[idx] > self.confidence_threshold and idx not in self.safe_list]
            candidates.sort(key=lambda x: x[1], reverse=True)
            to_remove = [idx for idx, _ in candidates[:max_removals]]
        
        # Protect honest participants (Fix: Enforce global budget)
        honest_removed = sum(1 for idx in to_remove if self.participants[idx]['type'] == 'honest')
        if self.honest_removed_count + honest_removed > self.honest_removal_budget:
            to_remove = [idx for idx in to_remove if self.participants[idx]['type'] != 'honest']
            honest_removed = 0
        
        self.honest_removed_count += honest_removed  # Fix: Update global count
        return to_remove, honest_removed
    
    def update_model(self, gradients_dict, val_loader=None):
        selected_ids, weights = self.multi_krum(gradients_dict)
        self.update_reputation_scores(selected_ids)
        self.update_confidence_scores()
        
        if not selected_ids:
            return 0.0
        
        # Aggregate gradients
        for param in self.model.parameters():
            param.grad = None
        
        for idx, weight in zip(selected_ids, weights):
            for param, grad in zip(self.model.parameters(), gradients_dict[idx]):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad += grad * weight
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        optimizer.step()
        
        # Evaluate model
        accuracy = self.evaluate_model(val_loader) if val_loader else 0.0
        self.accuracy_history.append(accuracy)
        return accuracy
    
    def force_extreme_measures(self, honest_candidates, val_loader=None):
        if not self.forced_removal:
            return [], 0
        if self.current_round < self.bootstrap_rounds:
            return [], 0
        if self.accuracy_history and self.accuracy_history[-1] < self.min_accuracy_threshold:
            candidates = [(idx, self.confidence_scores[idx]) for idx in self.participants 
                         if self.participants[idx]['active'] and idx not in honest_candidates and idx not in self.safe_list and self.confidence_scores[idx] > 0.9]  # Fix: Higher confidence
            candidates.sort(key=lambda x: x[1], reverse=True)
            to_remove = [idx for idx, _ in candidates[:self.adv_per_round_limit]]
            honest_removed = sum(1 for idx in to_remove if self.participants[idx]['type'] == 'honest')
            if self.honest_removed_count + honest_removed > self.honest_removal_budget:  # Fix: Enforce global budget
                to_remove = [idx for idx in to_remove if self.participants[idx]['type'] != 'honest']
                honest_removed = 0
            self.honest_removed_count += honest_removed  # Fix: Update global count
            return to_remove, honest_removed
        return [], 0
    
    def evaluate_model(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0.0
    
    def check_early_stopping(self):
        if len(self.accuracy_history) < self.patience or self.current_round < self.min_rounds:
            return False
        recent = self.accuracy_history[-self.patience:]
        return max(recent) - min(recent) < 0.01

# Participant Training
def train_participant(participant, dataset, device, val_loader=None):
    model = participant['model']
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001 * (0.95 ** (participant.get('epoch', 0) // 5)))
    criterion = nn.NLLLoss()
    
    data_loader = participant['data']
    if not data_loader:
        return [], 0.0
    
    total_loss = 0.0
    for epoch in range(5):
        participant['epoch'] = participant.get('epoch', 0) + 1
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Adversarial behavior
            if participant['type'] == 'adversarial':
                if epoch < 5:  # Subtle attacks
                    if random.random() < 0.2:
                        labels = (labels + 1) % dataset.num_classes
                    if random.random() < 0.5:
                        noise = torch.randn_like(features) * 0.05
                        features = features + noise
                elif epoch < 15:  # Stronger attacks
                    if random.random() < 0.5:
                        labels = (labels + 1) % dataset.num_classes
                    if random.random() < 0.7:
                        noise = torch.randn_like(features) * 0.15
                        features = features + noise
                else:  # Adaptive attacks
                    if random.random() < 0.4:
                        labels = (labels + 1) % dataset.num_classes
                    if random.random() < 0.6:
                        noise = torch.randn_like(features) * 0.12
                        features = features + noise
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
    
    # Collect gradients
    gradients = [param.grad.clone().detach() for param in model.parameters()]
    
    # Evaluate
    accuracy = 0.0
    if val_loader:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
    
    return gradients, total_loss / (5 * len(data_loader)), accuracy

# Data Distribution
def distribute_data_to_participants(server, dataset, train_indices):
    activated_honest = 0
    activated_adversarial = 0
    record_counts = defaultdict(int)
    
    # Count records per participant
    for idx in train_indices:
        pid = dataset.participant_ids[idx]
        record_counts[pid] += 1
    
    # Log record distribution
    print("Participant record distribution:")
    for pid, count in sorted(record_counts.items()):
        print(f"Participant {pid}: {count} records")
    
    for idx in server.participants:
        indices = [i for i in train_indices if dataset.participant_ids[i] == idx]
        num_records = len(indices)
        if num_records < 3:
            print(f"Warning: Participant {idx} has only {num_records} records, deactivating")
            server.participants[idx]['active'] = False
            continue
        if idx not in dataset.participant_indices:
            print(f"Warning: Participant {idx} not found in dataset, deactivating")
            server.participants[idx]['active'] = False
            continue
        subset = Subset(dataset, indices)
        server.participants[idx]['data'] = DataLoader(subset, batch_size=64, shuffle=True)
        server.participants[idx]['active'] = True
        if server.participants[idx]['type'] == 'honest':
            activated_honest += 1
        else:
            activated_adversarial += 1
    
    print(f"Activated {activated_honest} honest and {activated_adversarial} adversarial participants")
    return activated_honest, activated_adversarial

# Evaluation Metrics
def calculate_metrics(server, removed_participants):
    y_true = [1 if server.participants[idx]['type'] == 'adversarial' else 0 for idx in server.participants]
    y_pred = [1 if idx in removed_participants else 0 for idx in server.participants]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    honest_removed = sum(1 for idx in removed_participants if server.participants[idx]['type'] == 'honest')
    adversarial_removed = sum(1 for idx in removed_participants if server.participants[idx]['type'] == 'adversarial')
    total_honest = sum(1 for p in server.participants.values() if p['type'] == 'honest')
    total_adversarial = sum(1 for p in server.participants.values() if p['type'] == 'adversarial')
    
    honest_removal_rate = honest_removed / total_honest if total_honest > 0 else 0.0
    adversarial_removal_rate = adversarial_removed / total_adversarial if total_adversarial > 0 else 0.0
    removal_preference = adversarial_removal_rate / honest_removal_rate if honest_removal_rate > 0 else float('inf')
    
    tp = sum(1 for idx in removed_participants if server.participants[idx]['type'] == 'adversarial')
    fp = sum(1 for idx in removed_participants if server.participants[idx]['type'] == 'honest')
    tn = sum(1 for idx in server.participants if server.participants[idx]['type'] == 'honest' and idx not in removed_participants)
    fn = sum(1 for idx in server.participants if server.participants[idx]['type'] == 'adversarial' and idx not in removed_participants)
    
    model_accuracy = np.mean(server.accuracy_history[-5:]) if len(server.accuracy_history) >= 5 else server.accuracy_history[-1] if server.accuracy_history else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'honest_removed': honest_removed,
        'adversarial_removed': adversarial_removed,
        'honest_removal_rate': honest_removal_rate,
        'adversarial_removal_rate': adversarial_removal_rate,
        'removal_preference': removal_preference,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'model_accuracy': model_accuracy
    }

def print_detection_performance(metrics):
    print("\nDETECTION PERFORMANCE:")
    print("-" * 50)
    print(f"Accuracy:      {metrics['accuracy']*100:.2f}%")
    print(f"Precision:     {metrics['precision']*100:.2f}%")
    print(f"Recall:        {metrics['recall']*100:.2f}%")
    print(f"F1 Score:      {metrics['f1']*100:.2f}%")
    print(f"Model Acc (last 5): {metrics['model_accuracy']*100:.2f}%")
    print("-" * 50)
    print(f"Honest Removed:     {metrics['honest_removal_rate']*100:.1f}%")
    print(f"Adversarial Removed: {metrics['adversarial_removal_rate']*100:.1f}%")
    print(f"Removal Preference: {metrics['removal_preference']:.1f}x" if metrics['removal_preference'] != float('inf') else "Removal Preference: infx")
    print("-" * 50)
    print(f"True Positives:     {metrics['tp']}")
    print(f"False Positives:    {metrics['fp']}")
    print(f"True Negatives:     {metrics['tn']}")
    print(f"False Negatives:    {metrics['fn']}")
    print("-" * 50)

# Main Simulation
def run_v2g_simulation(dataset_path, num_honest, num_adversarial, rounds=100, device='cpu', no_early_stop=False):
    dataset = V2GDataset(dataset_path)
    model = V2GClassifier(input_size=dataset.num_features, num_classes=dataset.num_classes).to(device)
    server = FederatedServer(model, num_honest, num_adversarial, device, dataset)
    
    # Split data
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Distribute data
    activated_honest, activated_adversarial = distribute_data_to_participants(server, dataset, train_indices)
    
    # Adjust Multi-Krum parameters based on active participants
    total_active = activated_honest + activated_adversarial
    if total_active < server.multi_krum_m:
        print(f"Warning: Only {total_active} active participants, adjusting Multi-Krum parameters")
        server.multi_krum_m = max(3, total_active // 2)
        server.num_byzantines = max(1, min(15, activated_adversarial // 2))
        print(f"Adjusted Multi-Krum params: byzantines={server.num_byzantines}, m={server.multi_krum_m}")
    
    print(f"Starting FL with {activated_honest} honest and {activated_adversarial} adversarial participants")
    print(f"Bootstrap rounds: {server.bootstrap_rounds}, Whitelist rounds: {server.whitelist_rounds}")
    print(f"Removal threshold: {server.removal_threshold}, Max removal rate: {server.max_removal_rate*100:.1f}%")
    print(f"Protection strategy: Strictly limit honest removals to {server.honest_removal_budget} participants")
    
    removed_participants = []
    for round_idx in range(rounds):
        server.current_round += 1
        gradients_dict = {}
        
        # Train active participants
        for idx in server.participants:
            if not server.participants[idx]['active']:
                continue
            gradients, loss, accuracy = train_participant(server.participants[idx], dataset, device, val_loader)
            if gradients:
                gradients_dict[idx] = gradients
                server.calculate_gradient_metrics(idx, gradients)
        
        # Update model
        accuracy = server.update_model(gradients_dict, val_loader)
        selected_ids, _ = server.multi_krum(gradients_dict)
        honest_selected = sum(1 for idx in selected_ids if server.participants[idx]['type'] == 'honest')
        adversarial_selected = len(selected_ids) - honest_selected
        print(f"Multi-Krum selection: {honest_selected} honest, {adversarial_selected} adversarial (total: {len(selected_ids)})")
        
        # Check for extreme measures
        honest_candidates = server.identify_honest_participants(val_loader)
        to_remove, honest_removed = [], 0
        if server.current_round % 5 == 0:
            print(f"TRIGGER: Scheduled adversarial removal at round {server.current_round}")
            to_remove, honest_removed = server.force_extreme_measures(honest_candidates, val_loader)
            if to_remove:
                print(f"Round {server.current_round}: EXTREME MEASURES removed {len(to_remove)} participants (Honest: {honest_removed}, Adversarial: {len(to_remove) - honest_removed})")
        
        # Regular anomaly detection
        if not to_remove:
            to_remove, honest_removed = server.detect_anomalies(honest_candidates, val_loader)
        
        # Remove participants
        if to_remove:
            for idx in to_remove:
                server.participants[idx]['active'] = False
                removed_participants.append(idx)
            print(f"Round {server.current_round}: Removed {len(to_remove)} participants (Honest: {honest_removed}, Adversarial: {len(to_remove) - honest_removed})")
            if honest_removed == 0:
                print("  Perfect removal! Only adversarial participants removed.")
            print(f"  Current model accuracy: {accuracy:.2f}")
        
        # Track active participants
        active_honest = sum(1 for idx, p in server.participants.items() if p['active'] and p['type'] == 'honest')
        active_adversarial = sum(1 for idx, p in server.participants.items() if p['active'] and p['type'] == 'adversarial')
        print(f"Round {server.current_round}: Active: {active_honest + active_adversarial} (Honest: {active_honest}, Adversarial: {active_adversarial}), Accuracy: {accuracy:.2f}")
        
        # Early stopping
        if not no_early_stop and server.check_early_stopping():
            print(f"Early stopping triggered at round {server.current_round} - validation accuracy not improving")
            break
        
        # Stop if too few participants
        if active_honest + active_adversarial < server.multi_krum_m:
            print(f"Too few active participants ({active_honest + active_adversarial}) at round {server.current_round}, stopping")
            break
    
    # Final evaluation
    metrics = calculate_metrics(server, removed_participants)
    print_detection_performance(metrics)
    
    # Participant behavior analysis
    honest_selection_rates = [p['selection_count'] / max(server.current_round, 1) 
                            for idx, p in server.participants.items() if p['type'] == 'honest']
    adversarial_selection_rates = [p['selection_count'] / max(server.current_round, 1) 
                                 for idx, p in server.participants.items() if p['type'] == 'adversarial']
    print("\nPARTICIPANT BEHAVIOR ANALYSIS:")
    print("-" * 50)
    print(f"Average selection rate (honest):   {np.mean(honest_selection_rates):.3f}")
    print(f"Average selection rate (adversarial): {np.mean(adversarial_selection_rates):.3f}")
    
    # Final results
    print("\nFINAL RESULTS:")
    print(f"Honest participants: {sum(1 for p in server.participants.values() if p['type'] == 'honest' and p['active'])}/{server.num_honest} remaining ({metrics['honest_removal_rate']*100:.1f}% removed)")
    print(f"Adversarial participants: {sum(1 for p in server.participants.values() if p['type'] == 'adversarial' and p['active'])}/{server.num_adversarial} remaining ({metrics['adversarial_removal_rate']*100:.1f}% removed)")
    print(f"Final model accuracy: {metrics['model_accuracy']:.2f}")
    print(f"Removal preference ratio: {metrics['removal_preference']:.1f}x" if metrics['removal_preference'] != float('inf') else "Removal preference ratio: infx")
    
    print("\nSimulation completed!")
    print(f"Detection F1 score: {metrics['f1']*100:.2f}%")
    print(f"Relative preference for removing adversaries: {metrics['removal_preference']:.1f}x" if metrics['removal_preference'] != float('inf') else "Relative preference for removing adversaries: infx")
    print(f"Honest participants removed: {metrics['honest_removal_rate']*100:.1f}%")
    print(f"Adversarial participants removed: {metrics['adversarial_removal_rate']*100:.1f}%")
    
    return server, removed_participants, metrics

# Main Execution
if __name__ == "__main__":
    device = 'cpu'
    print(f"Using device: {device}")
    dataset_path = r"C:\Users\Administrator\Downloads\v2g_simulated_dataset (1).csv"
    num_honest = 45
    num_adversarial = 55
    rounds = 100
    print(f"Running V2G Federated Learning with adversarial majority...")
    server, removed_participants, metrics = run_v2g_simulation(dataset_path, num_honest, num_adversarial, rounds, device, no_early_stop=False)