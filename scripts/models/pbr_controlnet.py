"""
PBR Generation with ControlNet + Latent Code Conditioning
Using pretrained VAE weights:
1. VAE Encoder: 3-channel albedo input → latent (uses pretrained weights)
2. Diffusion: Latent diffusion + ControlNet(mask) + latent_code (cross-attention)
3. VAE Decoder: 3-channel albedo output (uses pretrained weights)
4. Separate heads: Extract intermediate features from VAE decoder to generate normal and displacement
"""
import torch
import torch.nn as nn
import einops
from submodule.controlnet.cldm.cldm import ControlLDM, ControlNet
from submodule.controlnet.ldm.models.diffusion.ddpm import LatentDiffusion
from submodule.controlnet.ldm.util import instantiate_from_config, default
from submodule.controlnet.ldm.modules.diffusionmodules.model import nonlinearity


class LatentCodeEmbedder(nn.Module):
    """Learnable style/material code embedder, replacing text embedding role"""
    def __init__(self, latent_dim=256, embed_dim=768, num_tokens=77):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        
        # MLP: latent_code -> text embedding
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * num_tokens)
        )
        
    def forward(self, latent_code):
        """
        Args:
            latent_code: [B, latent_dim]
        Returns:
            embedding: [B, num_tokens, embed_dim]
        """
        B = latent_code.shape[0]
        out = self.mlp(latent_code)  # [B, embed_dim * num_tokens]
        out = out.view(B, self.num_tokens, self.embed_dim)
        return out


class PBRDecoderHead(nn.Module):
    """
    Simple decoder head for normal or displacement.
    Takes intermediate features from VAE decoder (256x256) and outputs 512x512.
    """
    def __init__(self, in_channels=512, out_channels=3, target_resolution=512):
        super().__init__()
        # Input: 256x256 intermediate features
        # Output: 512x512 (one 2x upsampling)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 256 -> 512
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 4, out_channels, 3, padding=1)
        )
    
    def forward(self, x):
        return self.head(x)


class PBRControlLDM(ControlLDM):
    """
    PBR ControlNet using pretrained VAE weights:
    1. VAE Encoder: 3-channel albedo input → latent (pretrained)
    2. Diffusion: Latent diffusion + ControlNet(mask) + latent_code (cross-attention)
    3. VAE Decoder: 3-channel albedo output (pretrained)
    4. Separate heads: normal and displacement from VAE decoder intermediate features
    """
    def __init__(self, 
                 latent_code_dim=256,
                 use_latent_code=True,
                 loss_type='mixed',  # 'l1', 'l2', or 'mixed'
                 pixel_loss_weight=0.01,  # Weight for pixel-level PBR reconstruction losses
                 pixel_loss_frequency=10,  # Compute pixel loss every N steps (0 = disabled)
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_latent_code = use_latent_code
        self.loss_type = loss_type
        self.pixel_loss_weight = pixel_loss_weight
        self.pixel_loss_frequency = pixel_loss_frequency  # 0 = disabled, N = every N steps
        
        # Latent code embedder (learnable style/material code)
        if use_latent_code:
            context_dim = kwargs.get('unet_config', {}).get('params', {}).get('context_dim', 768)
            if isinstance(context_dim, list):
                context_dim = context_dim[0] if len(context_dim) > 0 else 768
            
            self.latent_embedder = LatentCodeEmbedder(
                latent_dim=latent_code_dim,
                embed_dim=context_dim,
                num_tokens=77
            )
        
        # Separate decoder heads for normal and displacement
        # These will be initialized after VAE decoder is created
        self.normal_head = None
        self.displacement_head = None
        self._heads_initialized = False
    
    def _init_pbr_heads(self):
        """Initialize separate decoder heads for normal and displacement"""
        if self._heads_initialized:
            return
        
        # Get intermediate feature channels from VAE decoder
        # Intermediate features are typically 512 channels at 256x256 resolution
        decoder = self.first_stage_model.decoder
        intermediate_channels = 512  # Standard for SD VAE: ch=128, ch_mult=[1,2,4,4], after 2nd upsampling
        
        # Create simple decoder heads
        self.normal_head = PBRDecoderHead(
            in_channels=intermediate_channels,
            out_channels=3,  # Normal: 3 channels
            target_resolution=512
        )
        self.displacement_head = PBRDecoderHead(
            in_channels=intermediate_channels,
            out_channels=1,  # Displacement: 1 channel
            target_resolution=512
        )
        
        # Initialize conv_out with pretrained weights if possible
        if hasattr(decoder, 'conv_out'):
            pretrained_conv_out = decoder.conv_out
            # Normal head: copy first 3 channels from pretrained
            if (self.normal_head.head[-1].in_channels == pretrained_conv_out.in_channels and
                pretrained_conv_out.out_channels >= 3):
                with torch.no_grad():
                    self.normal_head.head[-1].weight.data[:3] = pretrained_conv_out.weight.data[:3].clone()
                    if (self.normal_head.head[-1].bias is not None and 
                        pretrained_conv_out.bias is not None):
                        self.normal_head.head[-1].bias.data[:3] = pretrained_conv_out.bias.data[:3].clone()
            
            # Displacement head: copy first channel from pretrained
            if (self.displacement_head.head[-1].in_channels == pretrained_conv_out.in_channels and
                pretrained_conv_out.out_channels >= 1):
                with torch.no_grad():
                    self.displacement_head.head[-1].weight.data[0] = pretrained_conv_out.weight.data[0].clone()
                    if (self.displacement_head.head[-1].bias is not None and 
                        pretrained_conv_out.bias is not None):
                        self.displacement_head.head[-1].bias.data[0] = pretrained_conv_out.bias.data[0].clone()
        
        if hasattr(self, 'device'):
            self.normal_head.to(self.device)
            self.displacement_head.to(self.device)
        
        self._heads_initialized = True
    
    def decode_with_intermediate_features(self, z):
        """
        Decode VAE latent and extract intermediate features for separate heads.
        Uses pretrained VAE decoder for albedo, extracts intermediate features for normal/displacement.
        
        Returns: (albedo, intermediate_features)
        """
        z = self.first_stage_model.post_quant_conv(z)
        decoder = self.first_stage_model.decoder
        
        # Forward through decoder to get intermediate features
        temb = None
        h = decoder.conv_in(z)
        
        # Middle block
        h = decoder.mid.block_1(h, temb)
        h = decoder.mid.attn_1(h)
        h = decoder.mid.block_2(h, temb)
        
        # Upsampling - extract features after first upsampling level
        intermediate_features = None
        for i_level in reversed(range(decoder.num_resolutions)):
            for i_block in range(decoder.num_res_blocks + 1):
                h = decoder.up[i_level].block[i_block](h, temb)
                if len(decoder.up[i_level].attn) > 0:
                    h = decoder.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = decoder.up[i_level].upsample(h)
                # Extract features after first upsampling (i_level == decoder.num_resolutions - 2)
                if i_level == decoder.num_resolutions - 2:
                    intermediate_features = h
        
        # Final output (albedo) - uses pretrained decoder
        h = decoder.norm_out(h)
        h = nonlinearity(h)
        albedo = decoder.conv_out(h)
        if decoder.tanh_out:
            albedo = torch.tanh(albedo)
        
        # If intermediate_features is None, use the last feature before final conv
        if intermediate_features is None:
            intermediate_features = h
        
        return albedo, intermediate_features
    
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        """
        Override to support latent code conditioning.
        Use albedo (RGB) for encoding to leverage pretrained VAE weights.
        """
        # Ensure batch has 'txt' key (required by ControlNet)
        if 'txt' not in batch:
            batch['txt'] = [''] * len(batch[self.control_key])
        
        # Always use albedo (RGB) for encoding to use pretrained weights
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        z = x
        
        # Process control (hint/mask)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        
        # Latent code conditioning (learnable style/material code via cross-attention)
        if self.use_latent_code and 'latent_code' in batch:
            latent_code = batch['latent_code']
            if bs is not None:
                latent_code = latent_code[:bs]
            latent_code = latent_code.to(self.device)
            latent_embed = self.latent_embedder(latent_code)
            c = latent_embed
        else:
            # Use text conditioning
            c = self.get_learned_conditioning(batch.get('txt', [''] * len(control)))
            if bs is not None:
                c = c[:bs]
        
        # Pass PBR ground truth through cond for pixel-level loss computation (if enabled)
        cond_dict = dict(c_crossattn=[c], c_concat=[control])
        if 'pbr' in batch and self.pixel_loss_frequency > 0:
            pbr = batch['pbr']  # [B, H, W, 7]
            if bs is not None:
                pbr = pbr[:bs]
            pbr = pbr.to(self.device)
            cond_dict['pbr'] = pbr  # Pass through cond for loss computation
        
        return z, cond_dict
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """Standard diffusion model application with ControlNet"""
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, 
                                 control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), 
                                       timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, 
                                control=control, only_mid_control=self.only_mid_control)

        return eps
    
    def p_losses(self, x_start, cond, t, noise=None):
        """
        Override p_losses to compute losses.
        Standard latent space loss for diffusion training.
        Pixel-level losses removed to reduce memory usage.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # Latent space loss (noise prediction loss) - for diffusion training
        if self.loss_type == 'l1':
            loss_latent = (model_output - target).abs().mean([1, 2, 3])
        elif self.loss_type == 'l2':
            loss_latent = ((model_output - target) ** 2).mean([1, 2, 3])
        else:
            loss_latent = ((model_output - target) ** 2).mean([1, 2, 3])
        
        loss_dict.update({f'{prefix}/loss_latent': loss_latent.mean()})
        # gt
        albedo_gt = cond['pbr'].permute(0, 3, 1, 2)[:, :3, :, :]
        normal_gt = cond['pbr'].permute(0, 3, 1, 2)[:, 3:6, :, :]
        displacement_gt = cond['pbr'].permute(0, 3, 1, 2)[:, 6:7, :, :]
        # Initialize heads if needed
        if not self._heads_initialized:
            self._init_pbr_heads()
        
        # Compute pixel-level losses at reduced frequency to train decoder heads while saving memory
        loss_pixel_total = 0.0
        if (self.training and 
            self.pixel_loss_frequency > 0 and 
            'pbr' in cond):
            try:
                z_pred = 1. / self.scale_factor * model_output  # model_output: denoised z
                albedo_pred, normal_pred, displacement_pred = self.decode_pbr(z_pred)
                # L1 loss for all three maps (albedo, normal, displacement)
                loss_albedo = (albedo_pred - albedo_gt).abs().mean()
                loss_normal = (normal_pred - normal_gt).abs().mean()
                loss_displacement = (displacement_pred - displacement_gt).abs().mean()
                loss_pixel_total = loss_albedo + loss_normal + loss_displacement
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    loss_pixel_total = 0.0
                else:
                    raise e
        
        loss = loss_latent.mean() + self.pixel_loss_weight * loss_pixel_total
        loss_dict.update({f'{prefix}/loss_latent': loss_latent.mean()})
        loss_dict.update({f'{prefix}/loss_pixel': self.pixel_loss_weight * loss_pixel_total})
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict
    
    @torch.no_grad()
    def decode_pbr(self, z):
        """
        Decode latent to PBR maps.
        Uses pretrained VAE decoder for albedo, separate heads for normal/displacement.
        
        Args:
            z: [B, z_channels, H, W] - latent from diffusion model
        
        Returns:
            pbr: [B, 7, H_out, W_out] - concatenated [albedo(3), normal(3), displacement(1)]
        """
        if not self._heads_initialized:
            self._init_pbr_heads()
        
        # Decode to get albedo and intermediate features
        albedo, intermediate_features = self.decode_with_intermediate_features(z)
        
        # Generate normal and displacement from intermediate features
        normal = self.normal_head(intermediate_features)
        displacement = self.displacement_head(intermediate_features)
        
        return albedo, normal, displacement
    
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        """Override log_images method"""
        log = super().log_images(batch, N=N, n_row=n_row, sample=sample, ddim_steps=ddim_steps, 
                                 ddim_eta=ddim_eta, return_keys=return_keys, quantize_denoised=quantize_denoised,
                                 inpaint=inpaint, plot_denoise_rows=plot_denoise_rows, 
                                 plot_progressive_rows=plot_progressive_rows, plot_diffusion_rows=plot_diffusion_rows,
                                 unconditional_guidance_scale=unconditional_guidance_scale, 
                                 unconditional_guidance_label=unconditional_guidance_label,
                                 use_ema_scope=use_ema_scope, **kwargs)
        
        if 'latent_code' in batch:
            latent_info = [f"PBR Material\nLatent Code {i}" for i in range(min(N, len(batch.get('latent_code', []))))]
        else:
            latent_info = ["PBR Material\n(No Latent Code)"] * N
        
        from submodule.controlnet.ldm.util import log_txt_as_img
        log["conditioning"] = log_txt_as_img((512, 512), latent_info, size=16)
        
        return log
