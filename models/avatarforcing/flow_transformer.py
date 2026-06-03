import torch, os, math
import torch.nn as nn
import torch.nn.functional as F

from models import BaseModel
from timm.models.vision_transformer import Mlp
from timm.layers import use_fused_attn


def modulate(x, shift, scale) -> torch.Tensor:
    return x * (1 + scale) + shift

def enc_dec_mask(T, S, frame_width=1, expansion=2):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
    return mask == 0

def enc_dec_mask_window(start_pos: int, cache_len: int, n_tokens: int, expansion: int):
    q_abs = torch.arange(start_pos, start_pos + n_tokens)[:, None]
    k_abs = torch.arange(start_pos - cache_len, start_pos + n_tokens)[None, :]
    allow = (k_abs >= q_abs - expansion) & (k_abs <= q_abs + expansion)
    return allow

def make_block_bidir_causal_lookahead_mask(T: int, local_len: int, lookahead: int = 2):
    q = torch.arange(T)[:, None]
    k = torch.arange(T)[None, :]
    q_blk = q // local_len
    k_blk = k // local_len
    allow = (k_blk < q_blk) | (k <= q + lookahead)
    return allow

def make_block_bidir_causal_lookahead_mask_window(start_pos: int, cache_len: int, n_tokens: int, local_len: int, lookahead: int = 2):
    q_abs = torch.arange(start_pos, start_pos + n_tokens)[:, None]
    k_abs = torch.arange(start_pos - cache_len, start_pos + n_tokens)[None, :]
    q_blk = q_abs // local_len
    k_blk = k_abs // local_len
    allow = (k_blk < q_blk) | (k_abs <= q_abs + lookahead)
    return allow

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("RoPE dim must be even.")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))    
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()    
    return torch.polar(torch.ones_like(freqs), freqs) # [end, dim/2]


def apply_rotary_emb_causal(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor, 
    start_pos: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:

    B, H, N, D = xq.shape
    
    end_pos = start_pos + N
    if end_pos > freqs_cis.shape[0]:
        raise ValueError(f"Required position {end_pos} exceeds precomputed max length {freqs_cis.shape[0]}.")
        
    freqs_cis_n = freqs_cis[start_pos:end_pos, :].to(xq.device)
    freqs_cis_n = freqs_cis_n.view(1, 1, N, -1) # [1, 1, N, D/2]

    xq_ = xq.float().reshape(B, H, N, D // 2, 2)
    xk_ = xk.float().reshape(B, H, N, D // 2, 2)
    
    xq_comp = torch.view_as_complex(xq_)
    xk_comp = torch.view_as_complex(xk_)

    xq_rotated_comp = xq_comp * freqs_cis_n
    xk_rotated_comp = xk_comp * freqs_cis_n

    xq_out = torch.view_as_real(xq_rotated_comp).flatten(3)
    xk_out = torch.view_as_real(xk_rotated_comp).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_q(xq: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    B, H, N, D = xq.shape
    end_pos = start_pos + N
    freqs_cis_n = freqs_cis[start_pos:end_pos, :].view(1, 1, N, -1)
    xq_ = xq.float().reshape(B, H, N, D // 2, 2)
    xq_comp = torch.view_as_complex(xq_)
    xq_rotated_comp = xq_comp * freqs_cis_n
    xq_out = torch.view_as_real(xq_rotated_comp).flatten(3)
    return xq_out.type_as(xq)


def apply_rotary_emb_k(xk: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    B, H, N, D = xk.shape
    end_pos = start_pos + N
    freqs_cis_n = freqs_cis[start_pos:end_pos, :].view(1, 1, N, -1)
    xk_ = xk.float().reshape(B, H, N, D // 2, 2)
    xk_comp = torch.view_as_complex(xk_)
    xk_rotated_comp = xk_comp * freqs_cis_n
    xk_out = torch.view_as_real(xk_rotated_comp).flatten(3)
    return xk_out.type_as(xk)



class AudioMotionAttention(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.motion_embedder = nn.Sequential(
            nn.Linear(opt.dim_a, opt.dim_a),
            nn.LayerNorm(opt.dim_a),
            nn.SiLU(),
            nn.Linear(opt.dim_a, opt.dim_a),
            nn.LayerNorm(opt.dim_a),
            nn.SiLU(),
        )        

        self.dual_modal_attn = nn.MultiheadAttention(embed_dim=opt.dim_a, num_heads=4, batch_first=True)
        self.ln1 = nn.LayerNorm(opt.dim_a)
        self.silu1 = nn.SiLU()

        self.user_agent_attn = nn.MultiheadAttention(embed_dim=opt.dim_a, num_heads=4, batch_first=True)
        self.ln2 = nn.LayerNorm(opt.dim_a)
        self.silu2 = nn.SiLU()

        self.to_cond_user = nn.Linear(opt.dim_a, opt.dim_h)


    def forward(self, x, c, agent_audio):
        """
            x: audio (kv)
            c: motion (q)
            
        """
        c = self.motion_embedder(c)

        cross_modal_feat, _ = self.dual_modal_attn(c, x, x)
        cross_modal_feat = self.silu1(self.ln1(c + cross_modal_feat))

        unified, _ = self.user_agent_attn(agent_audio, cross_modal_feat, cross_modal_feat)
        unified = self.silu2(self.ln2(agent_audio + unified))
        unified = self.to_cond_user(unified)
        return unified


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int        = 8,
            qkv_bias: bool        = False,
            qk_norm: bool         = False,
            num_frames_for_clip   = 50,
            num_frames_per_block  = 10,
            attn_drop: float      = 0.,
            proj_drop: float      = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
        ) -> None:

        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.num_frames_for_clip = num_frames_for_clip
        self.num_frames_per_block = num_frames_per_block

        self.kv = nn.Linear(dim, dim * 2, bias = qkv_bias)
        self.q  = nn.Linear(dim, dim, bias = qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def cross_mask_local(q_len = 10, cache_len=2, expansion=2, device=None):
        k_len = cache_len + q_len  # 12
        q = torch.arange(q_len, device=device)[:, None]
        k = torch.arange(k_len, device=device)[None, :]
        center = q + cache_len                              
        mask = (k - center).abs() <= expansion              
        return mask


    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor = None, kv_cache = None, update_kv_cache=False, start_pos: int = 0) -> torch.Tensor:
        B, N, C = x.shape
        Nc = c.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)

        kv_new = self.kv(c).reshape(B, Nc, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_new, v_new = kv_new.unbind(0)
        k_new = self.k_norm(k_new)

        if kv_cache is not None: 
            if k_new.shape[2] == self.num_frames_per_block + 2:
                k_cache = kv_cache["k"]
                v_cache = kv_cache["v"]
                k_attn = torch.cat([k_cache, k_new], dim=2)
                v_attn = torch.cat([v_cache, v_new], dim=2) 
                mask = self.cross_mask_local(q_len = q.shape[2], cache_len=2, device=q.device)
            else:
                k_attn, v_attn = k_new, v_new

            if update_kv_cache:
                self.append_kv_to_buffer(kv_cache, k_new, v_new)

            x = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                attn_mask = mask,
                dropout_p = self.attn_drop.p if self.training else 0.)
        else:
            k_attn, v_attn = k_new, v_new
            x = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                attn_mask = mask,
                dropout_p = self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def append_kv_to_buffer(kv_cache, k_new, v_new):
        cache_size = kv_cache["k"].shape[2]
        num_new    = k_new.shape[2]
        kv_cache["k"] = k_new[:, :, -4:-2, :]
        kv_cache["v"] = v_new[:, :, -4:-2, :] #


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int        = 8,
            qkv_bias: bool        = False,
            qk_norm: bool         = False,
            num_frames_for_clip   = 50,
            num_frames_per_block  = 10,
            attn_drop: float      = 0.,
            proj_drop: float      = 0.,
            max_seq_len: int      = 1024,
            norm_layer: nn.Module = nn.LayerNorm,            
        ) -> None:

        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.num_frames_for_clip = num_frames_for_clip
        self.num_frames_per_block = num_frames_per_block

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, max_seq_len), persistent=False)
        self.apply_rotary_emb = apply_rotary_emb_causal


    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, start_pos:int = 0, kv_cache=None, update_kv_cache=False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv.unbind(0)

        q = self.q_norm(q) 
        k_new = self.k_norm(k_new)
        
        q = apply_rotary_emb_q(q, self.freqs_cis, start_pos=start_pos) # 12
        k_new = apply_rotary_emb_k(k_new, self.freqs_cis, start_pos=start_pos)

        if kv_cache is not None:
            if k_new.shape[2] == self.num_frames_per_block + 2:
                k_cache = kv_cache["k"] # 38
                v_cache = kv_cache["v"] # 38
                k_attn = torch.cat([k_cache, k_new], dim=-2)
                v_attn = torch.cat([v_cache, v_new], dim=-2)
                mask = mask[-(self.num_frames_per_block+2):, :]
            else:
                k_attn = k_new      # 50                                 
                v_attn = v_new      # 50
    
            if update_kv_cache:
                self.update_kv_cache(kv_cache, k_new, v_new)
    
            x = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                attn_mask = mask,
                dropout_p = self.attn_drop.p if self.training else 0.)
        else:
            k_attn, v_attn = k_new, v_new

            x = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                attn_mask = mask,
                dropout_p = self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def update_kv_cache(kv_cache, k_new, v_new):
        cache_size = kv_cache["k"].shape[2]
        num_new    = k_new.shape[2]
        if num_new >= cache_size:
            kv_cache["k"][:, :, :, :] = k_new[:, :, -cache_size - 2:-2, :]
            kv_cache["v"][:, :, :, :] = v_new[:, :, -cache_size - 2:-2, :]
        else:
            kv_cache["k"][:, :, :-10, :] = kv_cache["k"][:, :, 10:, :].clone()    # shift
            kv_cache["v"][:, :, :-10, :] = kv_cache["v"][:, :, 10:, :].clone()    # shift
            kv_cache["k"][:, :, -10:, :] = k_new[:, :, :-2, :]
            kv_cache["v"][:, :, -10:, :] = v_new[:, :, :-2, :]
        


class DFoTBlock(nn.Module):
    """
    A DFoT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, num_frames_for_clip, num_frames_per_block, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, num_frames_for_clip=num_frames_for_clip, num_frames_per_block=num_frames_per_block, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, num_frames_for_clip=num_frames_for_clip, num_frames_per_block=num_frames_per_block, **block_kwargs)

    def forward(self, x, scale_shift_alpha, c, mask=None, cross_mask=None, start_pos=0, kv_cache=None, update_kv_cache=False) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = scale_shift_alpha

        self_kv_cache  = kv_cache[0] if kv_cache is not None else None
        cross_kv_cache = kv_cache[1] if kv_cache is not None else None

        x_attn = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            mask            = mask,
            start_pos       = start_pos,
            kv_cache        = self_kv_cache,
            update_kv_cache = update_kv_cache)
        x = x + gate_msa * x_attn

        x_cross = self.cross_attn(
            self.norm3(x),
            c               = c,
            mask            = cross_mask,
            start_pos       = start_pos,
            kv_cache        = cross_kv_cache,
            update_kv_cache = update_kv_cache)

        x = x + x_cross
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class SequenceEmbed(nn.Module):
    def __init__(
            self,
            dim_w,
            dim_h,
            norm_layer = None,
            bias = True,
    ):
        super().__init__()
        self.proj = nn.Linear(dim_w, dim_h, bias=bias)
        self.norm = norm_layer(dim_h) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T = x.shape
        return self.norm(self.proj(x))


class FlowTransformer(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        self.num_frames_for_clip = int(self.opt.sec * self.opt.fps)
        self.num_frames_per_block = opt.num_frames_per_block

        self.hidden_size        = opt.dim_h
        self.num_heads          = opt.num_heads
        self.mlp_ratio          = opt.mlp_ratio
        self.transformer_depth  = opt.transformer_depth

        self.x_embedder = SequenceEmbed(opt.dim_w * 2, self.hidden_size)

        # optimal transport time encoding
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.c_embedder = AudioMotionAttention(opt)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True))

        self.blocks = nn.ModuleList([
            DFoTBlock(self.hidden_size, self.num_heads, mlp_ratio = self.mlp_ratio, num_frames_for_clip = self.num_frames_for_clip, num_frames_per_block = opt.num_frames_per_block)
                for _ in range(self.transformer_depth)])

        self.final_layer = FinalLayer(self.hidden_size, self.opt.dim_w)
        self.initialize_weights()
        
        # initialize the alignment mask
        alignment_mask = enc_dec_mask(self.num_frames_for_clip, self.num_frames_for_clip, 1, expansion=opt.attention_window).to(opt.rank)
        block_bidir_causal_lookahead_mask = make_block_bidir_causal_lookahead_mask(self.num_frames_for_clip, self.num_frames_per_block, 2).to(opt.rank)

        self.register_buffer('alignment_mask', alignment_mask)
        self.register_buffer('block_bidir_causal_lookahead_mask', block_bidir_causal_lookahead_mask)
        
        self._streaming_masks_cache = {}

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(
        self,
        t: torch.Tensor,      

        x: torch.Tensor, 
        wa: torch.Tensor,
        wa_user: torch.Tensor,
        motion_user: torch.Tensor,
        wr: torch.Tensor,
        train: bool     = True,

        start_pos       = 0,
        kv_cache_list   = None,
        update_kv_cache = False,
    ) -> torch.Tensor:
        """
        Forward pass of FlowMatchingTransformer.

        t:          (B,) tensor of starting timesteps in [0, 1]
        x:          (B, L, 512) : tensor of sequence of motion latent
        wa:  	    (B, L, 512)  / tensor sequence of wa latent 
        ref: 	    (B, 512)     / tensor of identity latent (i.e., s -> r)
        """
        t_embed = self.t_embedder(t.flatten()).unflatten(dim=0, sizes=t.shape)  	# N x D
        shared_scale_shift_alpha = self.adaLN_modulation(t_embed).chunk(6, dim=-1)

        wr = self.sequence_embedder(wr.unsqueeze(1), dropout_prob=self.opt.ref_dropout_prob, train=train)
        wr_embedded = wr.repeat(1, x.shape[1], 1)
        
        x = torch.cat([x, wr_embedded], dim=-1)
        x = self.x_embedder(x)

        wa          = self.sequence_embedder(wa, dropout_prob = self.opt.audio_dropout_prob, train=train)
        wa_user     = self.sequence_embedder(wa_user, dropout_prob = self.opt.audio_dropout_prob, train=train)        
        motion_user = self.sequence_embedder(motion_user, dropout_prob = self.opt.audio_dropout_prob, train=train)
        c           = self.c_embedder(wa_user, motion_user, wa)

        for i, block in enumerate(self.blocks):
            attn_kv_cache = kv_cache_list[i] if kv_cache_list is not None else None 
            x = block(
                x                 = x,
                scale_shift_alpha = shared_scale_shift_alpha,
                c                 = c,
                mask              = self.block_bidir_causal_lookahead_mask,
                cross_mask        = self.alignment_mask,
                start_pos         = start_pos,
                kv_cache          = attn_kv_cache,
                update_kv_cache   = update_kv_cache)  # (N, T, D)

        return self.final_layer(x, t_embed)


    def forward_with_precomputed(self, x, precomputed_c, precomputed_adaLN, precomputed_wr, start_pos, kv_cache_list, update_kv_cache):
        shared_scale_shift_alpha, t_embed, final_adaLN = precomputed_adaLN

        x = torch.cat([x, precomputed_wr], dim=-1)
        x = self.x_embedder(x)

        for i, block in enumerate(self.blocks):
            attn_kv_cache = kv_cache_list[i] if kv_cache_list is not None else None 
            x = block(
                x                 = x,
                scale_shift_alpha = shared_scale_shift_alpha,
                c                 = precomputed_c,
                mask              = self.block_bidir_causal_lookahead_mask,
                cross_mask        = self.alignment_mask,
                start_pos         = start_pos,
                kv_cache          = attn_kv_cache,
                update_kv_cache   = update_kv_cache)  # (N, T, D)

        return self.final_layer.forward_precomputed(x, final_adaLN)


    def sequence_embedder(self, sequence, dropout_prob, train=False) -> torch.Tensor:
        if train:
            batch_id_for_drop = torch.where(torch.rand(sequence.shape[0], device=sequence.device) < dropout_prob)
            sequence[batch_id_for_drop] = 0
        return sequence

    def compute_condition(self, wa: torch.Tensor, wa_user: torch.Tensor, motion_user: torch.Tensor, train: bool = False) -> torch.Tensor:
        wa          = self.sequence_embedder(wa, dropout_prob=self.opt.audio_dropout_prob, train=train)
        wa_user     = self.sequence_embedder(wa_user, dropout_prob=self.opt.audio_dropout_prob, train=train)
        motion_user = self.sequence_embedder(motion_user, dropout_prob=self.opt.audio_dropout_prob, train=train)
        return self.c_embedder(wa_user, motion_user, wa)


    def compute_wr_embedded(self, wr: torch.Tensor, seq_len: int, train: bool = False) -> torch.Tensor:
        wr = self.sequence_embedder(wr.unsqueeze(1), dropout_prob=self.opt.ref_dropout_prob, train=train)
        wr_embedded = wr.repeat(1, seq_len, 1)
        return wr_embedded


    def get_streaming_masks(
        self,
        start_pos: int,
        cache_len_self: int,
        cache_len_cross: int,
        n_tokens: int,
        device: torch.device
    ) -> tuple:
        """
        Get or create cached streaming attention masks.
        Masks depend on (start_pos % local_len, cache_lens, n_tokens) pattern.
        
        For streaming, the relative pattern repeats, so we cache by
        (start_pos modulo local_len) to maximize cache hits.
        """
        cache_key = (cache_len_self, cache_len_cross, n_tokens)
        
        if cache_key not in self._streaming_masks_cache:
            attn_mask = make_block_bidir_causal_lookahead_mask_window(
                start_pos=cache_len_self,  # Use cache_len as representative start
                cache_len=cache_len_self,
                n_tokens=n_tokens,
                local_len=self.num_frames_per_block,
                lookahead=2).to(device)
            
            if cache_len_cross > 0:
                cross_mask = enc_dec_mask_window(
                    start_pos=cache_len_self,
                    cache_len=cache_len_cross,
                    n_tokens=n_tokens,
                    expansion=self.opt.attention_window,
                ).to(device)
            else:
                cross_mask = None
            
            self._streaming_masks_cache[cache_key] = (attn_mask, cross_mask)
        
        return self._streaming_masks_cache[cache_key]


    def precompute_timestep_adaLN(
        self, 
        timestep_list: list, 
        batch_size: int, 
        seq_len: int, 
        device: torch.device,
        context_len: int = 0,
    ) -> list:
        nfe = len(timestep_list)
        all_t = torch.tensor(timestep_list, dtype=torch.float32, device=device)
        all_t_embed = self.t_embedder(all_t)

        all_adaLN = self.adaLN_modulation(all_t_embed)
        all_final_adaLN = self.final_layer.adaLN_modulation(all_t_embed)
        
        if context_len > 0:
            t_zero = torch.zeros(1, dtype=torch.float32, device=device)
            t_zero_embed = self.t_embedder(t_zero)
            adaLN_zero = self.adaLN_modulation(t_zero_embed)
            final_adaLN_zero = self.final_layer.adaLN_modulation(t_zero_embed)

            t_zero_embed_ctx = t_zero_embed.unsqueeze(0).expand(nfe, batch_size, -1).unsqueeze(2).expand(-1, -1, context_len, -1).contiguous()
            adaLN_zero_ctx = adaLN_zero.unsqueeze(0).expand(nfe, batch_size, -1).unsqueeze(2).expand(-1, -1, context_len, -1).contiguous()
            final_adaLN_zero_ctx = final_adaLN_zero.unsqueeze(0).expand(nfe, batch_size, -1).unsqueeze(2).expand(-1, -1, context_len, -1).contiguous()

            t_embed_denoise = all_t_embed.unsqueeze(1).unsqueeze(1).expand(nfe, batch_size, seq_len, -1).contiguous()
            adaLN_denoise = all_adaLN.unsqueeze(1).unsqueeze(1).expand(nfe, batch_size, seq_len, -1).contiguous()
            final_adaLN_denoise = all_final_adaLN.unsqueeze(1).unsqueeze(1).expand(nfe, batch_size, seq_len, -1).contiguous()

            all_t_embed_expanded = torch.cat([t_zero_embed_ctx, t_embed_denoise], dim=2)            
            all_adaLN_expanded = torch.cat([adaLN_zero_ctx, adaLN_denoise], dim=2)                  
            all_final_adaLN_expanded = torch.cat([final_adaLN_zero_ctx, final_adaLN_denoise], dim=2)
        else:
            all_t_embed_expanded = all_t_embed.unsqueeze(1).unsqueeze(1).expand(nfe, batch_size, seq_len, -1).contiguous()          # (nfe, B, L, hidden_size)        
            all_adaLN_expanded = all_adaLN.unsqueeze(1).unsqueeze(1).expand(nfe, batch_size, seq_len, -1).contiguous()              # (nfe, B, L, 6*hidden_size)        
            all_final_adaLN_expanded = all_final_adaLN.unsqueeze(1).unsqueeze(1).expand(nfe, batch_size, seq_len, -1).contiguous()  # (nfe, B, L, 2*hidden_size)
        
        precomputed = []
        for i in range(nfe):
            adaLN_chunked = all_adaLN_expanded[i].chunk(6, dim=-1)
            final_adaLN_chunked = all_final_adaLN_expanded[i].chunk(2, dim=-1)
            t_embed_i = all_t_embed_expanded[i]
            precomputed.append((adaLN_chunked, t_embed_i, final_adaLN_chunked))
        return precomputed



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of ConditionalFlowMatchingTransformer.
    """
    def __init__(self, hidden_size, dim_w):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, dim_w, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

    def forward_precomputed(self, x, precomputed_adaLN):
        """
        Forward with precomputed adaLN modulation (shift, scale).
        Avoids redundant adaLN_modulation computation.
        """
        shift, scale = precomputed_adaLN
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)
