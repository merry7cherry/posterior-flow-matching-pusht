"""Official two-frame image policy with a task-local PFM-Delta generator."""
import math
import torch
from torch import nn
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.pfm import delta_loss, log_variance, sample_delta


class PFMUnetHybridImagePolicy(DiffusionUnetHybridImagePolicy):
    def __init__(self, *, logvar_min, logvar_max, logvar_init, eps_t,
                 noise_scale, time_embed_scale, mixed_precision, sampling,
                 **kwargs):
        if not kwargs.get('obs_as_global_cond', True):
            raise ValueError('PFM supports the global condition protocol only')
        super().__init__(noise_scheduler=None, num_inference_steps=sampling['steps'], **kwargs)
        if self.horizon != 16 or self.action_dim != 2 or self.n_obs_steps != 2 or self.n_action_steps != 8:
            raise ValueError('expected official image PushT 16/2/2/8 dimensions')
        if not logvar_min < logvar_init < logvar_max or not 0 < eps_t < .5:
            raise ValueError('invalid PFM bounds')
        if noise_scale <= 0 or time_embed_scale <= 0 or mixed_precision not in ('bf16', 'fp32'):
            raise ValueError('invalid PFM scale or precision')
        self.logvar_min, self.logvar_max = logvar_min, logvar_max
        self.eps_t, self.noise_scale = eps_t, noise_scale
        self.time_embed_scale, self.mixed_precision = time_embed_scale, mixed_precision
        self.sampling = dict(sampling)
        head = nn.Conv1d(self.model.final_conv[1].in_channels, self.action_dim, 1)
        nn.init.zeros_(head.weight)
        prob = (logvar_init-logvar_min)/(logvar_max-logvar_min)
        nn.init.constant_(head.bias, math.log(prob/(1-prob)))
        self.model.variance_head = head
        self.loss_stats = {}
        self.sample_stats = {}
        # Optional visualization sink. Never serialized into model state.
        self.sample_trace = None
        self._sample_generators = None

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.device.type == 'cuda' and self.mixed_precision == 'bf16')

    def _field(self, x, t, condition):
        with self._autocast():
            return self.model(x, t.float() * self.time_embed_scale, global_cond=condition)

    def clip_gradients(self, max_norm):
        """Global gradient clipping for joint Gaussian NLL training."""
        norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm, error_if_nonfinite=True)
        return {'grad_norm': norm, 'grad_clip_scale': (max_norm / (norm + 1e-6)).clamp(max=1)}

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        x1 = self.normalizer['action'].normalize(batch['action']).float()
        images = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps].reshape(-1, *x.shape[2:]))
        with self._autocast():
            condition = self.obs_encoder(images).reshape(len(x1), -1)
        x0 = self.noise_scale * torch.randn_like(x1)
        t = self.eps_t + (1-2*self.eps_t) * torch.rand(len(x1), device=x1.device)
        xt = (1-t[:, None, None])*x0 + t[:, None, None]*x1
        v, raw = self._field(xt, t, condition)
        target = x1-x0
        ell = log_variance(raw, self.logvar_min, self.logvar_max)
        q = (target.float()-v.float()).square()
        loss = delta_loss(v, raw, target, self.logvar_min, self.logvar_max)
        with torch.no_grad():
            self.loss_stats = {
                'loss_total': loss.detach(),
                'velocity_mse': q.mean(),
                'velocity_variance': ell.exp().mean(),
                'logvar_mean': ell.mean(),
                'logvar_lower_saturation': (raw.float().sigmoid() < .01).float().mean(),
                'logvar_upper_saturation': (raw.float().sigmoid() > .99).float().mean()}
        return loss

    def reset(self):
        # Independent streams, continued across replans; repeatable runner resets.
        seed = int(self.sampling['seed'])
        self._sample_generators = [torch.Generator(device=self.device).manual_seed(seed+i)
                                   for i in range(3)]

    @torch.no_grad()
    def predict_action(self, obs_dict):
        # Parent owns normalization, frame order, action output and slicing.
        with self._autocast():
            return super().predict_action(obs_dict)

    def conditional_sample(self, condition_data, condition_mask, local_cond=None,
                           global_cond=None, generator=None, **kwargs):
        if local_cond is not None or bool(condition_mask.any()):
            raise ValueError('action inpainting is outside this PFM protocol')
        if self._sample_generators is None:
            self.reset()
        init_gen, brownian_gen, probe_gen = self._sample_generators
        initial = self.noise_scale * torch.randn(condition_data.shape, device=self.device,
                                                  generator=init_gen if generator is None else generator)
        condition = global_cond.detach()
        # FP32 probability math even when called under predict_action autocast.
        with torch.autocast(device_type=self.device.type, enabled=False):
            result, self.sample_stats = sample_delta(
                lambda x, t: self._field(x, t, condition), initial,
                steps=int(self.sampling['steps']), terminal_time=float(self.sampling['terminal_time']),
                solver=self.sampling['solver'], diffusion_scale=float(self.sampling['diffusion_scale']),
                probes=int(self.sampling['derivative_probes']), noise_scale=self.noise_scale,
                low=self.logvar_min, high=self.logvar_max,
                brownian_generator=brownian_gen, probe_generator=probe_gen,
                exact=self.sampling.get('exact_derivatives', False),
                trace=self.sample_trace)
        return result
