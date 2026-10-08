"""The generator: a text encoder, a style encoder over a writer's reference words, and a
denoising UNet (diffusers UNet2DConditionModel) on the canvas.

- Content: the word's characters (embedding, position, transformer) as cross-attention
  tokens, and its printed form (the prior) as a second input channel, stretched over the
  columns the handwriting should fill.
- Style: a CNN over each reference word (stride 16: 4 rows of tokens), a transformer over the
  tokens of all references together, giving cross-attention tokens (the details of the
  writer's letters) and a global style vector (added to the noise-level embedding), a width
  head (log of handwriting width over printed width) and a projection for a contrastive loss.
  The references are a set: nothing marks which reference a token came from, so any number of
  them can be given (training uses 1 to 4).
- UNet: the 64-row canvas is folded 2 x 2 into channels (pixel unshuffle) and denoised at
  32 rows by three levels (128, 256, 512 channels); attention at the two coarser levels.
- Classifier-free guidance: style replaced by learned null tokens, content by a NULL token
  and a blank prior.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from torch import nn

from .data import NULL_ID, NUM_CHARS


@dataclass
class ModelConfig:
    channels: tuple = (128, 256, 512)    # UNet levels at 32, 16 and 8 rows
    attn_from: int = 1                   # first level with attention
    layers_per_block: int = 2
    head_dim: int = 64                   # width of an attention head
    d: int = 384                         # width of the content and style tokens
    text_layers: int = 3
    style_layers: int = 2
    max_text: int = 48
    max_refs: int = 8                    # the most references the demo accepts (no parameters)
    null_tokens: int = 4


def sinusoid(n, d, device, scale=1.0):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None] * scale
    i = torch.arange(d // 2, device=device, dtype=torch.float32)[None]
    ang = pos / (10000 ** (2 * i / d))
    return torch.cat([ang.sin(), ang.cos()], 1)


def transformer(d, layers, heads):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class ContentEncoder(nn.Module):
    def __init__(self, d=384, layers=3, max_len=48):
        super().__init__()
        self.embed = nn.Embedding(NUM_CHARS + 2, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        self.encoder = transformer(d, layers, d // 64)
        self.norm = nn.LayerNorm(d)

    def forward(self, ids):
        """ids (B, T), 0 padding -> tokens (B, T, d), mask (B, T) True where a token is."""
        mask = ids != 0
        h = self.embed(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        return self.norm(self.encoder(h, src_key_padding_mask=~mask)), mask


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1)
        self.norm1 = nn.GroupNorm(min(32, cout // 4), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.norm2 = nn.GroupNorm(min(32, cout // 4), cout)
        self.skip = nn.Conv2d(cin, cout, 1, stride) if (cin != cout or stride != 1) else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        return F.silu(self.norm2(self.conv2(h)) + self.skip(x))


class StyleEncoder(nn.Module):
    def __init__(self, d=384, layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.SiLU(),                 # 32 rows
            ResBlock(32, 64), ResBlock(64, 64, 2),                # 16
            ResBlock(64, 128), ResBlock(128, 128, 2),             # 8
            ResBlock(128, 256), ResBlock(256, 256, 2),            # 4
            nn.Conv2d(256, d, 1))
        self.row = nn.Parameter(torch.zeros(4, d))
        self.encoder = transformer(d, layers, d // 64)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.width = nn.Linear(d, 1)
        self.proj = nn.Linear(d, 128)

    def forward(self, ref, ref_width, ref_mask):
        """ref (B, K, 64, R) in [-1, 1] (paper -1), ref_width (B, K) columns of each reference,
        ref_mask (B, K) which references there are -> tokens (B, K*4*R/16, d), their mask,
        global style (B, d)."""
        B, K, H, R = ref.shape
        f = self.cnn(ref.reshape(B * K, 1, H, R))                       # (B*K, d, 4, R/16)
        _, d, h, w = f.shape
        f = f + self.row[:h].t()[None, :, :, None] + sinusoid(w, d, f.device, 16.0).t()[None, :, None, :]
        tok = f.flatten(2).transpose(1, 2).reshape(B, K, h * w, d)
        cols = torch.arange(w, device=ref.device)
        valid = (cols[None, None, :] < (ref_width[:, :, None] + 15) // 16) & ref_mask[:, :, None]   # (B, K, w)
        valid = valid[:, :, None, :].expand(B, K, h, w).reshape(B, K * h * w)
        tok = self.norm(self.encoder(tok.reshape(B, K * h * w, d), src_key_padding_mask=~valid))
        m = valid[..., None].to(tok.dtype)
        g = self.head((tok * m).sum(1) / m.sum(1).clamp(min=1))
        return tok, valid, g


class Denoiser(nn.Module):
    """F(x, prior; c_noise, tokens, global style) on the 64-row canvas, run at 32 rows."""

    def __init__(self, cfg):
        super().__init__()
        n = len(cfg.channels)
        down = tuple("CrossAttnDownBlock2D" if i >= cfg.attn_from else "DownBlock2D" for i in range(n))
        up = tuple("CrossAttnUpBlock2D" if i >= cfg.attn_from else "UpBlock2D" for i in reversed(range(n)))
        self.unet = UNet2DConditionModel(
            sample_size=None, in_channels=8, out_channels=4, down_block_types=down, up_block_types=up,
            block_out_channels=tuple(cfg.channels), layers_per_block=cfg.layers_per_block,
            # diffusers' `attention_head_dim` is, for historical reasons, the number of heads
            cross_attention_dim=cfg.d, attention_head_dim=tuple(max(c // cfg.head_dim, 1) for c in cfg.channels),
            norm_num_groups=32,
            class_embed_type="projection", projection_class_embeddings_input_dim=cfg.d,
            resnet_time_scale_shift="scale_shift")

    def forward(self, x, prior, c_noise, tokens, mask, g):
        h = F.pixel_unshuffle(torch.cat([x, prior], 1), 2)
        y = self.unet(h, c_noise, encoder_hidden_states=tokens, encoder_attention_mask=mask, class_labels=g).sample
        return F.pixel_shuffle(y, 2)


class Generator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.content = ContentEncoder(c.d, c.text_layers, c.max_text)
        self.style = StyleEncoder(c.d, c.style_layers)
        self.denoiser = Denoiser(c)
        self.null_tokens = nn.Parameter(torch.randn(c.null_tokens, c.d) * 0.02)
        self.null_style = nn.Parameter(torch.zeros(c.d))

    def condition(self, text, ref, ref_width, ref_mask, drop_style=None, drop_content=None):
        """-> dict(tokens, mask, g, g_ref) for the denoiser. drop_style / drop_content: (B,) bool,
        replace the style by the null style / the text by NULL (the prior is blanked by the
        caller). g_ref is the references' style before any dropping."""
        B = text.shape[0]
        if drop_content is not None and drop_content.any():
            null = torch.zeros_like(text)
            null[:, 0] = NULL_ID
            text = torch.where(drop_content[:, None], null, text)
        ct, cm = self.content(text)
        st, sm, g = self.style(ref, ref_width, ref_mask)
        g_ref = g
        if drop_style is not None and drop_style.any():
            n = self.cfg.null_tokens
            nt = torch.zeros_like(st)
            nt[:, :n] = self.null_tokens.to(st.dtype)
            nm = torch.zeros_like(sm)
            nm[:, :n] = True
            st = torch.where(drop_style[:, None, None], nt, st)
            sm = torch.where(drop_style[:, None], nm, sm)
            g = torch.where(drop_style[:, None], self.null_style.to(g.dtype).expand(B, -1), g)
        return {"tokens": torch.cat([ct, st], 1), "mask": torch.cat([cm, sm], 1), "g": g, "g_ref": g_ref}

    def forward(self, x, prior, c_noise, cond):
        return self.denoiser(x, prior, c_noise, cond["tokens"], cond["mask"].to(x.dtype), cond["g"])


def supcon(z, labels, tau=0.1):
    """Supervised contrastive loss (Khosla et al. 2020) over L2-normalised z (N, k)."""
    z = F.normalize(z.float(), dim=1)
    sim = z @ z.t() / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    per = -(logp * pos).sum(1) / pos.sum(1).clamp(min=1)
    return per[pos.any(1)].mean()


def config_dict(cfg):
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}


def count(module):
    return sum(p.numel() for p in module.parameters())
