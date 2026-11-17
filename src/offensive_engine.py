import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BiologicalDecayModule(nn.Module):
    """
    Simulates biological degradation using Langevin Dynamics (for physical tremors)
    and ACT-R architecture (for non-linear cognitive fatigue).
    """
    def __init__(self, mu: float = 0.0, sigma: float = 0.05, act_r_d: float = 0.1):
        super().__init__()
        # These could be learnable parameters for the attacker, but we start with static
        self.mu = nn.Parameter(torch.tensor(mu))
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.d = nn.Parameter(torch.tensor(act_r_d))

    def inject_langevin_noise(self, spatial_tensor: torch.Tensor) -> torch.Tensor:
        """
        Injects stochastic micro-tremors into spatial coordinates.
        dx_t = mu(x_t, t)dt + sigma * dW_t
        We approximate this by adding Wiener Brownian noise.
        """
        if spatial_tensor.numel() == 0:
            return spatial_tensor
            
        dt = 1.0 # arbitrary time step
        # Brownian noise: N(0, dt)
        dW = torch.randn_like(spatial_tensor) * math.sqrt(dt)
        
        # Drift component: mu * x_t * dt (simple linear drift)
        drift = self.mu * spatial_tensor * dt
        
        # Diffusion component: sigma * dW
        diffusion = self.sigma * dW
        
        return spatial_tensor + drift + diffusion

    def inject_actr_fatigue(self, timing_tensor: torch.Tensor) -> torch.Tensor:
        """
        Injects cognitive fatigue using ACT-R activation decay.
        A_i = ln(sum(t_j - d)) where d is decay parameter.
        We simulate this by increasing flight times (index 1) exponentially over sequence length.
        """
        if timing_tensor.numel() == 0:
            return timing_tensor
            
        B, seq_len, feats = timing_tensor.shape
        
        # Create a time step index (j) normalized from 0 to 1
        t_j = torch.linspace(0, 1, seq_len, device=timing_tensor.device).unsqueeze(0).unsqueeze(-1)
        
        # Fatigue factor increases non-linearly over time
        # Applying A_i = exp(t_j * d) - 1 to scale the flight times
        # Higher 'd' means faster fatigue
        fatigue_multiplier = torch.exp(t_j * self.d)
        
        perturbed_tensor = timing_tensor.clone()
        # Assume feature 0 is hold time, feature 1 is flight time
        # Fatigue mostly affects flight time (hesitation) and slightly affects hold time
        if feats > 0:
            perturbed_tensor[:, :, 0] = perturbed_tensor[:, :, 0] * (1.0 + (fatigue_multiplier.squeeze(-1) - 1.0) * 0.2)
        if feats > 1:
            perturbed_tensor[:, :, 1] = perturbed_tensor[:, :, 1] * fatigue_multiplier.squeeze(-1)
            
        return perturbed_tensor

    def forward(self, batch: dict) -> dict:
        """
        Applies biological decay to the batch.
        """
        out_batch = batch.copy()
        
        if 'mouse_spatial' in out_batch and out_batch['mouse_spatial'] is not None:
            out_batch['mouse_spatial'] = self.inject_langevin_noise(out_batch['mouse_spatial'])
            
        if 'keystroke_timing' in out_batch and out_batch['keystroke_timing'] is not None:
            out_batch['keystroke_timing'] = self.inject_actr_fatigue(out_batch['keystroke_timing'])
            
        return out_batch

class ReplayAttackModule(nn.Module):
    """
    Simulates a fast-replay attack where the attacker plays back stolen biometric telemetry at a faster speed.
    """
    def __init__(self, speedup_factor: float = 1.5):
        super().__init__()
        self.speedup_factor = speedup_factor

    def forward(self, batch: dict) -> dict:
        out_batch = batch.copy()
        if 'keystroke_timing' in out_batch and out_batch['keystroke_timing'] is not None:
            # Speeds up timing (shorter hold/flight times)
            out_batch['keystroke_timing'] = out_batch['keystroke_timing'] / self.speedup_factor
            
        if 'mouse_kinematics' in out_batch and out_batch['mouse_kinematics'] is not None:
            # Speeds up velocity
            out_batch['mouse_kinematics'] = out_batch['mouse_kinematics'] * self.speedup_factor
            
        return out_batch

class RandomNoiseAttackModule(nn.Module):
    """
    Simulates a naive attacker injecting uniform random noise into the spatial trajectory.
    """
    def __init__(self, noise_scale: float = 0.5):
        super().__init__()
        self.noise_scale = noise_scale

    def forward(self, batch: dict) -> dict:
        out_batch = batch.copy()
        if 'mouse_spatial' in out_batch and out_batch['mouse_spatial'] is not None:
            noise = torch.randn_like(out_batch['mouse_spatial']) * self.noise_scale
            out_batch['mouse_spatial'] = out_batch['mouse_spatial'] + noise
        return out_batch

class JitterAttackModule(nn.Module):
    """
    Simulates a hardware glitch or a shaky mouse sensor (high frequency, low amplitude noise).
    """
    def __init__(self, jitter_intensity: float = 0.1):
        super().__init__()
        self.jitter_intensity = jitter_intensity

    def forward(self, batch: dict) -> dict:
        out_batch = batch.copy()
        if 'mouse_kinematics' in out_batch and out_batch['mouse_kinematics'] is not None:
            # Apply rapid jitter to velocity, but keep spatial mostly intact
            jitter = (torch.rand_like(out_batch['mouse_kinematics']) - 0.5) * self.jitter_intensity
            out_batch['mouse_kinematics'] = out_batch['mouse_kinematics'] + jitter
        return out_batch

class MidSessionHijackModule(nn.Module):
    """
    Simulates a mid-session hijack where the first half of the data is genuine (clean),
    and the second half is replaced by a biological deepfake.
    """
    def __init__(self, deepfake_generator: nn.Module):
        super().__init__()
        self.deepfake_generator = deepfake_generator

    def forward(self, batch: dict) -> dict:
        hijacked_batch = self.deepfake_generator(batch)
        out_batch = batch.copy()
        
        # Splice genuine data and fake data
        if 'mouse_spatial' in out_batch and out_batch['mouse_spatial'] is not None:
            seq_len = out_batch['mouse_spatial'].shape[1]
            midpoint = seq_len // 2
            out_batch['mouse_spatial'][:, midpoint:] = hijacked_batch['mouse_spatial'][:, midpoint:]
            
        if 'mouse_kinematics' in out_batch and out_batch['mouse_kinematics'] is not None:
            seq_len = out_batch['mouse_kinematics'].shape[1]
            midpoint = seq_len // 2
            out_batch['mouse_kinematics'][:, midpoint:] = hijacked_batch['mouse_kinematics'][:, midpoint:]
            
        return out_batch

class HybridGenerator(nn.Module):
    """
    Hybrid GAN-VAE architecture to generate synchronized mouse and keystroke sequences.
    """
    def __init__(self, latent_dim: int = 128, mouse_seq_len: int = 2048, key_seq_len: int = 500):
        super().__init__()
        self.latent_dim = latent_dim
        self.mouse_seq_len = mouse_seq_len
        self.key_seq_len = key_seq_len
        
        # Shared Decoder base
        self.fc = nn.Linear(latent_dim, 256)
        self.relu = nn.ReLU()
        
        # Modality-specific heads
        # Mouse (spatial: 2, kinematics: 2) -> 4 features
        self.mouse_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, mouse_seq_len * 4)
        )
        
        # Keystroke (hold: 1, flight: 1) -> 2 features
        self.key_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, key_seq_len * 2)
        )

    def forward(self, z: torch.Tensor):
        x = self.relu(self.fc(z))
        
        mouse_flat = self.mouse_head(x)
        key_flat = self.key_head(x)
        
        mouse_seq = mouse_flat.view(-1, self.mouse_seq_len, 4)
        key_seq = key_flat.view(-1, self.key_seq_len, 2)
        
        mouse_spatial = mouse_seq[:, :, 0:2]
        mouse_kinematics = mouse_seq[:, :, 2:4]
        
        return mouse_spatial, mouse_kinematics, key_seq

class CompositeLoss(nn.Module):
    """
    L_hybrid = alpha * L_GAN + beta * L_VAE + gamma * L_sync + delta * L_grad
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, delta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, fake_logits, vae_recon_loss, vae_kl_loss, mouse_features, key_features):
        # Adversarial Loss (Discriminator fooled)
        l_gan = self.bce(fake_logits, torch.ones_like(fake_logits))
        
        # VAE Loss
        l_vae = vae_recon_loss + vae_kl_loss
        
        # Sync Penalty (ensure mouse and key features are roughly temporally aligned in latent space)
        # Using a simple cosine similarity as a proxy for temporal sync
        m_mean = mouse_features.mean(dim=1)
        k_mean = key_features.mean(dim=1)
        l_sync = 1.0 - F.cosine_similarity(m_mean, k_mean).mean()
        
        # Gradient Penalty (for WGAN-GP stability)
        l_grad = 0.0 # Placeholder for WGAN gradient penalty
        
        total_loss = (self.alpha * l_gan) + (self.beta * l_vae) + (self.gamma * l_sync) + (self.delta * l_grad)
        return total_loss

if __name__ == '__main__':
    # Automated Tests
    print("Testing BiologicalDecayModule...")
    decay = BiologicalDecayModule(mu=0.01, sigma=0.1, act_r_d=0.2)
    
    mock_batch = {
        'mouse_spatial': torch.zeros(2, 100, 2), # flat trajectory
        'keystroke_timing': torch.ones(2, 50, 2) # constant timing
    }
    
    perturbed = decay(mock_batch)
    
    # Assert Langevin added noise (variance > 0)
    assert perturbed['mouse_spatial'].std() > 0.05, "Langevin dynamics failed to inject sufficient noise."
    
    # Assert ACT-R increased the flight time at the end of the sequence vs the beginning
    t_start = perturbed['keystroke_timing'][0, 0, 1].item()
    t_end = perturbed['keystroke_timing'][0, -1, 1].item()
    assert t_end > t_start, f"ACT-R fatigue failed: start={t_start}, end={t_end}"
    
    print("BiologicalDecayModule tests passed!")
    
    print("Testing HybridGenerator...")
    gen = HybridGenerator(latent_dim=64, mouse_seq_len=100, key_seq_len=50)
    z = torch.randn(4, 64)
    m_s, m_k, k_t = gen(z)
    
    assert m_s.shape == (4, 100, 2)
    assert m_k.shape == (4, 100, 2)
    assert k_t.shape == (4, 50, 2)
    
    print("HybridGenerator tests passed!")
