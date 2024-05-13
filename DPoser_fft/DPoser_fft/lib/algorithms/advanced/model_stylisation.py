import functools

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

    return sigmas


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class ComplexAct(nn.Module):
    def __init__(self, act, use_phase=False):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexAct, self).__init__()
        self.act = act
        self.use_phase = use_phase

    def forward(self, z):
        if self.use_phase:
            return self.act(torch.tanh(torch.abs(z))) * torch.exp(1.j * torch.angle(z)) 
        else:
            return self.act(z.real) + 1.j * self.act(z.imag)

class ComplexGnorm(nn.Module):
    def __init__(self, norm):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexGnorm, self).__init__()
        self.norm = norm

    def forward(self, z):
        return self.norm(z.real) + 1.j * self.norm(z.imag)

def get_act(config,c):
    """Get activation functions from the config file."""
    if c:
        if config.model.nonlinearity.lower() == 'elu':
            return ComplexAct(nn.ELU())
        elif config.model.nonlinearity.lower() == 'relu':
            return ComplexAct(nn.ReLU())
        elif config.model.nonlinearity.lower() == 'lrelu':
            return ComplexAct(nn.LeakyReLU(negative_slope=0.2))
        elif config.model.nonlinearity.lower() == 'swish':
            return ComplexAct(nn.SiLU())
        else:
            raise NotImplementedError('activation function does not exist!')
    else:
        if config.model.nonlinearity.lower() == 'elu':
            return nn.ELU()
        elif config.model.nonlinearity.lower() == 'relu':
            return nn.ReLU()
        elif config.model.nonlinearity.lower() == 'lrelu':
            return nn.LeakyReLU(negative_slope=0.2)
        elif config.model.nonlinearity.lower() == 'swish':
            return nn.SiLU()
        else:
            raise NotImplementedError('activation function does not exist!')


class TimeMLPs(torch.nn.Module):
    def __init__(self, config, n_poses=21, pose_dim=6, hidden_dim=64, n_blocks=2):
        super().__init__()
        dim = n_poses * pose_dim
        self.act = get_act(config)

        layers = [torch.nn.Linear(dim + 1, hidden_dim),
                  self.act]

        for _ in range(n_blocks):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                self.act,
                torch.nn.Dropout(p=config.model.dropout)
            ])

        layers.append(torch.nn.Linear(hidden_dim, dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, t, condition=None, mask=None):
        return self.net(torch.cat([x, t[:, None]], dim=1))

def get_dropout(p):
    def dropout_complex(x):
        # work around unimplemented dropout for complex
        if x.is_complex():
            mask = torch.nn.functional.dropout(torch.ones_like(x.real),p=p)
            return x * mask
        else:
            return torch.nn.functional.dropout(x,p=p)

    return dropout_complex

    

class ScoreModelFC(nn.Module):
    """
    Independent condition feature projection layers for each block
    """

    def __init__(self, config, n_poses=21, pose_dim=6, hidden_dim=64,
                 embed_dim=32, n_blocks=2):
        super(ScoreModelFC, self).__init__()

        self.config = config
        self.n_poses = n_poses
        self.joint_dim = pose_dim
        self.n_blocks = n_blocks

        self.act = get_act(config,True)
        # self.actr = get_act(config,False)

        self.pre_dense = nn.Linear((21//2+1)*3, hidden_dim,dtype=torch.cfloat)
        self.pre_dense_t = nn.Linear(embed_dim, hidden_dim,dtype=torch.cfloat)
        self.pre_dense_cond = nn.Linear(hidden_dim, hidden_dim,dtype=torch.cfloat)
        self.pre_gnorm = ComplexGnorm(nn.GroupNorm(32, num_channels=hidden_dim))
        # self.dropout = nn.Dropout(p=config.model.dropout)
        # self.dropout = get_dropout(p=config.model.dropout)

        # self.pre_denser = nn.Linear(n_poses * pose_dim, hidden_dim)
        # self.pre_dense_tr = nn.Linear(embed_dim, hidden_dim)
        # self.pre_dense_condr = nn.Linear(hidden_dim, hidden_dim)
        # self.pre_gnormr = nn.GroupNorm(32, num_channels=hidden_dim)
        # self.dropout = nn.Dropout(p=config.model.dropout)
        # self.dropoutr = nn.Dropout(p=config.model.dropout)
        self.dropout = get_dropout(p=config.model.dropout)

        #complex weight
        self.complex_weight = nn.Parameter(torch.randn(21//2+1,2,dtype=torch.float32) * 0.02)

        # time embedding
        self.time_embedding_type = config.model.embedding_type.lower()
        if self.time_embedding_type == 'fourier':
            self.gauss_proj = GaussianFourierProjection(embed_dim=hidden_dim, scale=config.model.fourier_scale)
        elif self.time_embedding_type == 'positional':
            self.posit_proj = functools.partial(get_timestep_embedding, embedding_dim=hidden_dim)
        else:
            assert 0

        self.shared_time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim,dtype=torch.cfloat),
            self.act,
        )

        self.act =  ComplexAct(nn.SiLU())
        self.emblayers = nn.Linear(hidden_dim, 2 * hidden_dim, dtype=torch.cfloat)
        self.norm = ComplexGnorm(nn.LayerNorm(hidden_dim))
        self.outlayers= nn.Linear(hidden_dim, hidden_dim, dtype=torch.cfloat)


        self.register_buffer('sigmas', torch.tensor(get_sigmas(config), dtype=torch.float))

        for idx in range(n_blocks):
            setattr(self, f'b{idx + 1}_emblayers1', nn.Sequential(ComplexAct(nn.SiLU()),nn.Linear(hidden_dim, 2 * hidden_dim,dtype=torch.cfloat)))
            setattr(self, f'b{idx + 1}_norm1', ComplexGnorm(nn.LayerNorm(hidden_dim)))
            setattr(self, f'b{idx + 1}_act1', ComplexAct(nn.SiLU()))
            setattr(self, f'b{idx + 1}_out_layers1', nn.Linear(hidden_dim, hidden_dim,dtype=torch.cfloat))

            setattr(self, f'b{idx + 1}_emblayers2', nn.Sequential(ComplexAct(nn.SiLU()),nn.Linear(hidden_dim, 2 * hidden_dim,dtype=torch.cfloat)))
            setattr(self, f'b{idx + 1}_norm2', ComplexGnorm(nn.LayerNorm(hidden_dim)))
            setattr(self, f'b{idx + 1}_act2', ComplexAct(nn.SiLU()))
            setattr(self, f'b{idx + 1}_out_layers2', nn.Linear(hidden_dim, hidden_dim,dtype=torch.cfloat))
        

        self.post_dense = nn.Linear(hidden_dim, (21//2+1)*3,dtype=torch.cfloat)
        # self.post_denser = nn.Linear(hidden_dim, n_poses * pose_dim)

    def forward(self, batch, t, condition=None, mask=None, train=True):
        """
        batch: [B, j*3] or [B, j*6]
        t: [B]
        Return: [B, j*3] or [B, j*6] same dim as batch
        """
        B = batch.shape[0]
        bs = batch.shape[0]

        # batch = batch.view(bs, -1)  # [B, j*3]

        # time embedding
        if self.time_embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = self.gauss_proj(torch.log(used_sigmas))
        elif self.time_embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            used_sigmas = self.sigmas[t.long()]
            temb = self.posit_proj(timesteps)
        else:
            raise ValueError(f'time embedding type {self.time_embedding_type} unknown.')

        B,D = temb.shape
        # tembr=temb
        # tembr = self.shared_time_embedr(tembr)
        
        extend = torch.zeros_like(temb)
        temb = torch.cat([temb,extend],-1)
        temb = temb.reshape(B,D,2)
        temb = torch.view_as_complex(temb)
        temb = self.shared_time_embed(temb)

        h = self.pre_dense(batch)
        # h += self.pre_dense_t(temb)
        # h = self.pre_gnorm(h)
        # h = self.act(h)
        # h = self.dropout(h)

        temb = self.act(temb)
        emb_out = self.emblayers(temb)
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = self.norm(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        h = self.outlayers(h)

        for idx in range(self.n_blocks):

            emb_out = getattr(self, f'b{idx + 1}_emblayers1')(temb)
            scale, shift = torch.chunk(emb_out, 2, dim=-1)
            h1 = getattr(self, f'b{idx + 1}_norm1')(h) * (1 + scale) + shift
            h1 = getattr(self, f'b{idx + 1}_act1')(h1)
            h1 = self.dropout(h1)
            h1 = getattr(self, f'b{idx + 1}_out_layers1')(h1)

            
            emb_out = getattr(self, f'b{idx + 1}_emblayers2')(temb)
            scale, shift = torch.chunk(emb_out, 2, dim=-1)
            h2 = getattr(self, f'b{idx + 1}_norm2')(h1) * (1 + scale) + shift
            h2 = getattr(self, f'b{idx + 1}_act2')(h2)
            h2 = self.dropout(h2)
            h2 = getattr(self, f'b{idx + 1}_out_layers2')(h2)

            h = h + h2


        res = self.post_dense(h)  # [B, j*3]

        # res = res.reshape(B,21//2+1,3)
        # res = res.transpose(1,2)
        # res = res*torch.view_as_complex(self.complex_weight)
        # res = res.transpose(2,1)
        # res = res.reshape(B,11*3)

        res = res.reshape(B,21//2+1,3)
        res = res.transpose(1,2)
        res = res*torch.view_as_complex(self.complex_weight)+res
        # res = torch.fft.irfft(res,dim=-1,n=21)
        res = res.transpose(2,1)
        res = res.reshape(B,33)

        #result is complex

        ''' normalize the output '''
        if self.config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((bs, 1))
            #cast into complex
            # used_sigmas = torch.complex(used_sigmas, torch.zeros_like(used_sigmas))
            res = res / used_sigmas

        return res