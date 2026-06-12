import math
import torch
from torch import nn
import torch.nn.functional as F
from rl_games.algos_torch.network_builder import NetworkBuilder


class RnnDecTestNet(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):

        print("\n\n\n\n\n\n RnnDecTestNet initialized with params: ", params, "\n\n\n\n\n\n")
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.obs_size_ = 13
        num_inputs = input_shape[0]  # Assuming input_shape is a tuple or list with the first element as the number of inputs
        self.num_inputs = num_inputs
        print("num_inputs: ", num_inputs)
        self.actions_num = actions_num
        print('RnnDecTestNet num_inputs, actions_num: ', num_inputs, actions_num)

        self.width = width = 50 # 50 #50 #50
        self.gru_width = 50 #50
        self.central_value = params.get('central_value', False)
        self.value_size = kwargs.pop('value_size', 1)
        self.linear1 = nn.Linear(self.obs_size_, width)
        self.linear2 = nn.Linear(width, width)

        if num_inputs > self.obs_size_:
            self.linear_1 = nn.Linear(num_inputs - 13, width)
            self.gru = nn.GRU(2*width, self.gru_width, bidirectional=False,  batch_first=True)
        else:
            self.gru = nn.GRU(width, self.gru_width, bidirectional=False,  batch_first=True)


        self.mean_linear = nn.Linear(self.gru_width, actions_num)
        self.logstd_linear = nn.Linear(self.gru_width, actions_num)  # Assuming logstd is needed
        self.value_linear = nn.Linear(self.gru_width, 1)

    def is_rnn(self):
        return True
    
    def get_default_rnn_state(self):
        if not self.training:
            return [torch.zeros(1, 2560, self.gru_width)]  #20480 for evaluation (num_layers * num_directions, batch, hidden_size)
        return [torch.zeros(1, 2048, self.gru_width)]  #20480 for evaluation (num_layers * num_directions, batch, hidden_size)

    def forward(self, obs):
        
        observ = obs['obs']
        hidden = obs['rnn_states'][-1]
        seq_len = obs.get('seq_length', 1) # Default to 1 if not provided
        
        batch_size = obs['rnn_states'][0].shape[1] #2048
        observ = observ.view(batch_size, seq_len, -1)  # (2048, 8, 16) or (2048, 1, 16)
        obs, side_obs = torch.split(observ, [13, observ.shape[-1]-13], dim=-1)
        x = F.elu(self.linear1(obs))  
        x = F.elu(self.linear2(x))
        u = x.clone()
        if observ.shape[-1] > self.obs_size_:
            x_ = F.elu(self.linear_1(side_obs))
            x = torch.cat([x, x_], dim=-1)
            
        x, h = self.gru(x, hidden) # h: (num_layers * num_directions, batch, hidden_size)
        x = x + u # Residual connection around GRU
        x = x.reshape(batch_size*seq_len, -1)  # Flatten back to (batch_size * seq_length, hidden_size)
       
        action = self.mean_linear(x)
        logstd = self.logstd_linear(x)  
        value = self.value_linear(x)
        if self.central_value:
            return value, None
        return action, logstd, value, [h], None 



class RnnTestNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return RnnDecTestNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)
 

  
class DecTestNet(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):

        print("\n DecTestNet initialized with params \n")
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        num_inputs = input_shape[0]  # Assuming input_shape is a tuple or list with the first element as the number of inputs
        width = 100 
        if num_inputs == 67 or num_inputs == 19:
            self.obs_width = 19
        else:
            self.obs_width = 13
        
        # Check if we have side observations (embodiment info)
        self.has_side_obs = num_inputs > self.obs_width
        self.obs_width, self.has_side_obs = 17, False
        self.central_value = params.get('central_value', False)
        self.value_size = kwargs.pop('value_size', 1)
        self.linear1 = nn.Linear(self.obs_width, width) #18, width)
        self.linear2 = nn.Linear(width, width)
        
        if self.has_side_obs:
            self.linear_1 = nn.Linear(num_inputs-self.obs_width, width)
            self.linear3 = nn.Linear(2*width, 2*width)
            self.norm = nn.LayerNorm(2 * width)
        else:
            self.linear3 = nn.Linear(width, 2*width)
            self.norm = nn.LayerNorm(width)
            
        self.linear4 = nn.Linear(2*width, width)
        self.mean_linear = nn.Linear(width, actions_num) #nn.Linear(width, actions_num)
        self.logstd_linear = nn.Linear(width, actions_num) #nn.Linear(width, actions_num)  # Assuming logstd is needed
        self.value_linear = nn.Linear(width, 1) #nn.Linear(width, 1)

    def is_rnn(self):
        return False

    def forward(self, obs):
        obs = obs['obs']
        if obs.shape[0] == 1:
             obs = obs.squeeze(0)       
        
        if self.has_side_obs:
            # Process with embodiment information
            obs, side_obs = torch.split(obs, [self.obs_width, obs.shape[-1]-self.obs_width], dim=-1)
            
            x = F.elu(self.linear1(obs))  
            x = F.elu(self.linear2(x))
            x_ = F.elu(self.linear_1(side_obs))
            x = torch.cat([x, x_], dim=-1)
            x = self.norm(x)
            x = F.elu(self.linear3(x))
            x = F.elu(self.linear4(x))
        else:
            # Process without embodiment information (scenario 1 case)
            x = F.elu(self.linear1(obs))  
            x = F.elu(self.linear2(x))
            x = self.norm(x)
            x = F.elu(self.linear3(x))
            x = F.elu(self.linear4(x))
        
        action = self.mean_linear(x)
        logstd = self.logstd_linear(x)  
        value = self.value_linear(x)
        if self.central_value:
            return value, None
        return action , logstd, value, None, x.detach()

    
    def forward_wm(self, obs):
        obs = obs['obs']
        
        if obs.shape[0] == 1:
             obs = obs.squeeze(0)
        obs, side_obs = torch.split(obs, [self.obs_width, obs.shape[-1]-self.obs_width], dim=-1)
        
        x = F.elu(self.linear1(obs))  
        x = F.elu(self.linear2(x))
        x_ = F.elu(self.linear_1(side_obs))
        x = torch.cat([x, x_], dim=-1)
        x = self.norm(x)
        x = F.elu(self.linear3(x))
        x = F.elu(self.linear4(x))

        return x
    
class DecTestNet_Hyper(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):

        print("\n DecTestNet_Hyper initialized with params \n")
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        num_inputs = input_shape[0]  # Assuming input_shape is a tuple or list with the first element as the number of inputs
        width = 64
        self.central_value = params.get('central_value', False)
        self.value_size = kwargs.pop('value_size', 1)
        self.linear1 = nn.Linear(13, width)
        self.linear2 = nn.Linear(width, width)
        self.hyper = HyperLinear(in_dim=width, out_dim=width, cond_dim=num_inputs-13, hidden_dim=128)
        self.linear3 = nn.Linear(width, width)
        self.linear4 = nn.Linear(width, width)
        self.mean_linear = nn.Linear(width, actions_num)
        self.logstd_linear = nn.Linear(width, actions_num)  # Assuming logstd is needed
        self.value_linear = nn.Linear(width, 1)

    def is_rnn(self):
        return False

    def forward(self, obs):
        obs = obs['obs']
        if obs.shape[0] == 1:
             obs = obs.squeeze(0)
        obs, side_obs = torch.split(obs, [13, obs.shape[-1]-13], dim=-1)
        
        x = F.elu(self.linear1(obs))  
        x = F.elu(self.hyper(x, side_obs))
        x = F.elu(self.linear3(x))
        x = F.elu(self.linear4(x))

        action = self.mean_linear(x)
        logstd = self.logstd_linear(x)  # Assuming you want to output logstd as well
        logstd = torch.nan_to_num(logstd, nan=0.0, posinf=0.0, neginf=0.0)
        value = self.value_linear(x)
        if self.central_value:
            return value, None
        return action, logstd, value, None

class DecTestNet_film(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):

        print("\n DecTestNet initialized with params \n")
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        num_inputs = input_shape[0]  # Assuming input_shape is a tuple or list with the first element as the number of inputs
        width = 64
        self.central_value = params.get('central_value', False)
        self.value_size = kwargs.pop('value_size', 1)

        # FiLM layer
        self.film = FiLM(cond_dim=num_inputs-13, num_channels=width)
        self.linear1 = nn.Linear(13, width)
        self.linear2 = nn.Linear(width, width)
        self.linear3 = nn.Linear(width, width)
        self.linear4 = nn.Linear(width, width)
        self.mean_linear = nn.Linear(width, actions_num)
        self.logstd_linear = nn.Linear(width, actions_num)  # Assuming logstd is needed
        self.value_linear = nn.Linear(width, 1)

    def is_rnn(self):
        return False

    def forward(self, obs):
        obs = obs['obs']
        if obs.shape[0] == 1:
             obs = obs.squeeze(0)
        obs, side_obs = torch.split(obs, [13, obs.shape[-1]-13], dim=-1)
        
        x = F.elu(self.linear1(obs))  
        x = F.elu(self.linear2(x))
        x = F.elu(self.film(x, side_obs))
        x = F.elu(self.linear3(x))
        x = F.elu(self.linear4(x))

        action = self.mean_linear(x)
        logstd = self.logstd_linear(x)  # Assuming you want to output logstd as well
        logstd = torch.nan_to_num(logstd, nan=0.0, posinf=0.0, neginf=0.0)
        value = self.value_linear(x)
        if self.central_value:
            return value, None
        return action, logstd, value, None



class DecTestNet_MTRL(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        print("\n DecTestNet initialized with params \n")
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        num_inputs = input_shape[0]
        width = 64
        self.central_value = params.get('central_value', False)
        self.value_size = kwargs.pop('value_size', 1)

        # ===== Custom parameters =====
        self.k = params.get('k', 4)  # number of encoders
        num_heads = params.get('num_heads', 1)

        # ===== Encoders for the first part =====
        self.first_part_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(13, width),
                nn.ELU(),
                nn.Linear(width, width)
            ) for _ in range(self.k)
        ])

        # ===== Side observation encoder =====
        self.side_encoder = nn.Linear(num_inputs - 13, width)

        # ===== Attention mechanism =====
        self.attn = nn.MultiheadAttention(embed_dim=width, num_heads=num_heads, batch_first=True)

        # ===== Post-attention layers =====
        self.linear3 = nn.Linear(2 * width, 2 * width)
        self.linear4 = nn.Linear(2 * width, width)

        # ===== Output heads =====
        self.mean_linear = nn.Linear(width, actions_num)
        self.logstd_linear = nn.Linear(width, actions_num)
        self.value_linear = nn.Linear(width, 1)

    def is_rnn(self):
        return False

    def forward(self, obs):
        obs = obs['obs']
        if obs.shape[0] == 1:
            obs = obs.squeeze(0)
        obs_first, side_obs = torch.split(obs, [13, obs.shape[-1] - 13], dim=-1)

        # Encode first part with k encoders
        encoded_list = [F.elu(enc(obs_first)) for enc in self.first_part_encoders]
        encoded_stack = torch.stack(encoded_list, dim=-2)  # [B, k, width]

        # Encode side_obs
        side_embed = F.elu(self.side_encoder(side_obs)).unsqueeze(-2)  # [B, 1, width]

        # Attention: query=side_obs, key/value=encoded first parts
        attn_out, _ = self.attn(query=side_embed, key=encoded_stack, value=encoded_stack)
        attn_out = attn_out.squeeze(-2)  # [B, width]
        side_embed = side_embed.squeeze(-2)  # [B, width]

        # Concatenate attention output and side embedding
        x = torch.cat([attn_out, side_embed], dim=-1)
        x = F.elu(self.linear3(x))
        x = F.elu(self.linear4(x))

        # Output heads
        action = self.mean_linear(x)
        logstd = self.logstd_linear(x)
        value = self.value_linear(x)

        if self.central_value:
            return value, None
        return action, logstd, value, None


class DecTestNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return DecTestNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)
    



class BatchedSpecialists(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)
        #super().__init__()
        self.E = params.get('E', 256) # Changed from 2 to 256 to match expected embodiment count
        print(f"[NETWORK INIT] BatchedSpecialists initialized with E={self.E} embodiments")
        self.obs_dim = 17
        self.act_dim = 4
        self.hidden = 100
        
        # Per-embodiment observation normalization parameters
        # obs normalization per teacher
        self.register_buffer("obs_mean", torch.zeros(self.E, self.obs_dim))
        self.register_buffer("obs_var", torch.ones(self.E, self.obs_dim))
        self.register_buffer("obs_count", torch.zeros(self.E))

        # value normalization if you need it
        self.register_buffer("value_mean", torch.zeros(self.E, 1))
        self.register_buffer("value_var", torch.ones(self.E, 1))
        self.register_buffer("value_count", torch.zeros(self.E))
        
        # Add step counter for debugging
        self.step_count = 0
        self.last_emb_stats = None
        # Layer 1 weights for each embodiment 
        self.W1 = nn.Parameter(torch.randn(self.E, self.hidden, self.obs_dim) * (1/math.sqrt(self.obs_dim))) 
        self.b1 = nn.Parameter(torch.zeros(self.E, self.hidden)) 
        # Layer 2 
        self.W2 = nn.Parameter(torch.randn(self.E, self.hidden, self.hidden) * (1/math.sqrt(self.hidden))) 
        self.b2 = nn.Parameter(torch.zeros(self.E, self.hidden))

        self.W3 = nn.Parameter(torch.randn(self.E, 2*self.hidden, self.hidden) * (1/math.sqrt(self.hidden)))
        self.b3 = nn.Parameter(torch.zeros(self.E, 2*self.hidden))

        self.W4 = nn.Parameter(torch.randn(self.E, self.hidden, 2*self.hidden) * (1/math.sqrt(2*self.hidden)))
        self.b4 = nn.Parameter(torch.zeros(self.E, self.hidden))

        self.Wa = nn.Parameter(torch.randn(self.E, self.act_dim, self.hidden) * (1/math.sqrt(self.hidden)))
        self.ba = nn.Parameter(torch.zeros(self.E, self.act_dim))

        self.Wv = nn.Parameter(torch.randn(self.E, 1, self.hidden) * (1/math.sqrt(self.hidden)))
        self.bv = nn.Parameter(torch.zeros(self.E, 1))

        self.Wstd = nn.Parameter(torch.randn(self.E, self.act_dim, self.hidden) * (1/math.sqrt(self.hidden)))
        self.bstd = nn.Parameter(torch.zeros(self.E, self.act_dim))

        self.norm_weight = nn.Parameter(torch.ones(self.E, self.hidden))
        self.norm_bias = nn.Parameter(torch.zeros(self.E, self.hidden))

    def _layernorm(self, x, emb_id, eps=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + eps)
        return x * self.norm_weight[emb_id] + self.norm_bias[emb_id]

    def _layer(self, x, emb_id, W, b):
        # x: [B, in]
        # W: [E, out, in], b: [E, out]
        Wb = W[emb_id]                  # [B, out, in]
        bb = b[emb_id]                  # [B, out]
        y = torch.bmm(Wb, x.unsqueeze(-1)).squeeze(-1) + bb
        return y
    
    def is_rnn(self):
        return False

    def forward(self, input_dict):
        #print('batched specialists', self.E)
        obs = input_dict["obs"]         # [B, obs_dim], float
        if obs.shape[0] == 1:
             obs = obs.squeeze(0)
        emb_id = torch.floor(self.E*input_dict["emb_ids"]).long()   # [B], long in [0..E-1]

        # Normalize observations per embodiment
        obs = (obs - self.obs_mean[emb_id]) / torch.sqrt(self.obs_var[emb_id] + 1e-8)        
        # Per-embodiment normalization disabled for performance
        # Individual training still works through separate network parameters per embodiment
        
        # Debugging disabled to prevent torch.compile graph breaks
        # Individual training verification shows it's working correctly:
        # - Different parameters between embodiments ✓  
        # - Different actions for same observation ✓
        # self.step_count += 1
        
        x = F.elu(self._layer(obs, emb_id, self.W1, self.b1))
        x = F.elu(self._layer(x, emb_id, self.W2, self.b2))

        # per-embodiment LayerNorm
        x = self._layernorm(x, emb_id)

        x = F.elu(self._layer(x, emb_id, self.W3, self.b3))  # 100 -> 200
        x = F.elu(self._layer(x, emb_id, self.W4, self.b4))  # 200 -> 100

        mu = self._layer(x, emb_id, self.Wa, self.ba)
        logstd = self._layer(x, emb_id, self.Wstd, self.bstd)
        v = self._layer(x, emb_id, self.Wv, self.bv)
        
        # Per-embodiment value normalization disabled for performance
        
        return mu, logstd, v, None, None


class ParTestNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return BatchedSpecialists(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)




class FiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(cond_dim, 2*num_channels),
            nn.ReLU(),
            nn.Linear(2*num_channels, 2*num_channels)
        )
        # init so gamma≈1, beta≈0
        #nn.init.zeros_(self.gen[-1].weight)
        #nn.init.zeros_(self.gen[-1].bias)
        self.num_channels = num_channels

    def forward(self, x, z):
        # x: (B,C,H,W), z: (B,cond_dim)
        gb = self.gen(z)                           # (B, 2C)
        gamma, beta = gb.chunk(2, dim=1)          # (B,C), (B,C)
        #gamma = gamma.unsqueeze(-1).unsqueeze(-1) # (B,C,1,1)
        #beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class HyperLinear(nn.Module):
    """
    A linear layer whose weights (and biases) are generated by a hypernetwork.
    """
    def __init__(self, in_dim, out_dim, cond_dim, hidden_dim=128):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Hypernetwork to generate weight & bias
        self.hyper = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (in_dim * out_dim) + out_dim)
        )
    
    def forward(self, x, cond):
        """
        x: [B, in_dim]
        cond: [B, cond_dim]
        """
        # Generate dynamic weights and biases
        params = self.hyper(cond)
        W, b = params[:, :self.in_dim * self.out_dim], params[:, self.in_dim * self.out_dim:]
        W = W.view(-1, self.out_dim, self.in_dim)
        
        # Batched linear: (B,out_dim,in_dim) x (B,in_dim,1)
        out = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1) + b
        return out
