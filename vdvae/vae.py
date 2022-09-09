import torch
from torch import nn
from torch.nn import functional as F
from .vae_helpers import (
    HModule,
    get_1x1,
    get_3x3,
    DmolNet,
    draw_gaussian_diag_samples,
    gaussian_analytical_kl,
)
from collections import defaultdict
import numpy as np
import itertools


class Block(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=None,
        residual=False,
        use_3x3=True,
        zero_last=False,
    ):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c3 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


def parse_layer_string(s):
    layers = []
    for ss in s.split(","):
        if "x" in ss:
            res, num = ss.split("x")
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif "m" in ss:
            res, mixin = [int(a) for a in ss.split("m")]
            layers.append((res, mixin))
        elif "d" in ss:
            res, down_rate = [int(a) for a in ss.split("d")]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(",")
        for ss in s:
            k, v = ss.split(":")
            mapping[int(k)] = int(v)
    return mapping


class Encoder(HModule):
    def build(self):
        H = self.H
        self.in_conv = get_3x3(H.image_channels, H.width)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(H.enc_blocks)
        for res, down_rate in blockstr:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(
                Block(
                    self.widths[res],
                    int(self.widths[res] * H.bottleneck_multiple),
                    self.widths[res],
                    down_rate=down_rate,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2).contiguous()
        x = x.contiguous()
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            x = block(x)
            res = x.shape[2]
            x = (
                x
                if x.shape[1] == self.widths[res]
                else pad_channels(x, self.widths[res])
            )
            activations[res] = x
        return activations


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim
        self.enc = Block(
            width * 2, cond_width, H.zdim * 2, residual=False, use_3x3=use_3x3
        )
        self.prior = Block(
            width,
            cond_width,
            H.zdim * 2 + width,
            residual=False,
            use_3x3=use_3x3,
            zero_last=True,
        )
        self.z_proj = get_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        # self.z_fn = lambda x: self.z_proj(x)
        # self.z_fn = self.z_proj

    def sample_manual_epsilon(self, x, acts, eps):
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        pm, pv, xpp = (
            feats[:, : self.zdim, ...],
            feats[:, self.zdim : self.zdim * 2, ...],
            feats[:, self.zdim * 2 :, ...],
        )
        x = x + xpp

        sigma = F.elu(qv) + 1
        z = sigma * eps + qm

        pz = torch.linalg.norm((z - qm) / torch.sqrt(sigma)) ** 2
        return z, x, pz.detach()

    def sample_uncond_manual_epsilon(self, x, eps):
        feats = self.prior(x)
        pm, pv, xpp = (
            feats[:, : self.zdim, ...],
            feats[:, self.zdim : self.zdim * 2, ...],
            feats[:, self.zdim * 2 :, ...],
        )
        x = x + xpp

        sigma = F.elu(pv) + 1
        z = sigma * eps + pm

        pz = torch.linalg.norm((z - pm) / torch.sqrt(sigma)) ** 2
        return z, x, pz.detach()

    def sample(self, x, acts):
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        pm, pv, xpp = (
            feats[:, : self.zdim, ...],
            feats[:, self.zdim : self.zdim * 2, ...],
            feats[:, self.zdim * 2 :, ...],
        )
        x = x + xpp
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, x, kl

    def sample_with_aug(self, x, acts, aug_model, layer):
        qmv = self.enc(torch.cat([x, acts], dim=1))
        qm, qv = aug_model(qmv, layer).chunk(2, dim=1)

        feats = self.prior(x)
        xpp = feats[:, self.zdim * 2 :, ...]
        x = x + xpp
        z = draw_gaussian_diag_samples(qm, qv)
        return z, x

    def sample_uncond(self, x, t=None, lvs=None):
        n, c, h, w = x.shape
        feats = self.prior(x)
        pm, pv, xpp = (
            feats[:, : self.zdim, ...],
            feats[:, self.zdim : self.zdim * 2, ...],
            feats[:, self.zdim * 2 :, ...],
        )
        x = x + xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def forward_manual_epsilon(self, xs, activations, epsilon):
        if activations is None:
            try:
                x = xs[self.base]
            except KeyError:
                ref = xs[list(xs.keys())[0]]
                x = torch.zeros(
                    dtype=ref.dtype,
                    size=(ref.shape[0], self.widths[self.base], self.base, self.base),
                    device=ref.device,
                )
            z, x, pz = self.sample_uncond_manual_epsilon(x, epsilon)
        else:
            x, acts = self.get_inputs(xs, activations)
            if self.mixin is not None:
                x = x + F.interpolate(
                    xs[self.mixin][:, : x.shape[1], ...],
                    scale_factor=self.base // self.mixin,
                )
            z, x, pz = self.sample_manual_epsilon(x, acts, epsilon)
        # x = x + self.z_fn(z)
        x = x + self.z_proj(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs, pz, z.detach()

    def forward(self, xs, activations, get_latents=False):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(
                xs[self.mixin][:, : x.shape[1], ...],
                scale_factor=self.base // self.mixin,
            )
        z, x, kl = self.sample(x, acts)
        # x = x + self.z_fn(z)
        x = x + self.z_proj(z)
        x = self.resnet(x)
        xs[self.base] = x
        if get_latents:
            return xs, dict(z=z.detach(), kl=kl)
        return xs, dict(kl=kl)

    def forward_with_aug(self, xs, activations, aug_model, layer):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(
                xs[self.mixin][:, : x.shape[1], ...],
                scale_factor=self.base // self.mixin,
            )
        z, x = self.sample_with_aug(x, acts, aug_model, layer)

        # x = x + self.z_fn(z)
        x = x + self.z_proj(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs

    def forward_uncond(self, xs, t=None, lvs=None):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(
                dtype=ref.dtype,
                size=(ref.shape[0], self.widths[self.base], self.base, self.base),
                device=ref.device,
            )
        if self.mixin is not None:
            x = x + F.interpolate(
                xs[self.mixin][:, : x.shape[1], ...],
                scale_factor=self.base // self.mixin,
            )
        z, x = self.sample_uncond(x, t, lvs=lvs)
        # x = x + self.z_fn(z)
        x = x + self.z_proj(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs


class Decoder(HModule):
    def build(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)

        for i, res in enumerate(self.resolutions):
            if res <= H.no_bias_above:
                parameter = nn.Parameter(torch.zeros(1, self.widths[res], res, res))
                setattr(self, f"bias_xs_{i}", parameter)
                # instead of ParameterList (to support DataParallel)
        self.len_bias_list = i
        # self.bias_xs = nn.ParameterList(
        #     [
        #         nn.Parameter(torch.zeros(1, self.widths[res], res, res))
        #         for res in self.resolutions
        #         if res <= H.no_bias_above
        #     ]
        # )
        self.out_net = DmolNet(H)
        self.gain = nn.Parameter(torch.ones(1, H.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.width, 1, 1))
        # self.final_fn = lambda x: x * self.gain + self.bias

    def final_fn(self, x):
        return x * self.gain + self.bias

    def forward_manual_epsilon(self, activations, epsilons, start=0):
        stats = []
        xs = {}
        for i in range(self.len_bias_list):
            b = getattr(self, f"bias_xs_{i}")
            xs[b.shape[2]] = b
        # xs = {a.shape[2]: a for a in self.bias_xs}
        for l, (block, eps) in enumerate(
            itertools.zip_longest(self.dec_blocks, epsilons)
        ):
            if l < start:
                xs, _ = block.forward(xs, activations)
            else:
                xs, pz, latents = block.forward_manual_epsilon(xs, activations, eps)
                stats.append({"pz": pz, "latents": latents})
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats

    def forward(self, activations, get_latents=False):
        stats = []
        xs = {}
        for i in range(self.len_bias_list):
            b = getattr(self, f"bias_xs_{i}")
            xs[b.shape[2]] = b
        # xs = {a.shape[2]: a for a in self.bias_xs}
        for block in self.dec_blocks:
            xs, block_stats = block(xs, activations, get_latents=get_latents)
            stats.append(block_stats)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats

    def forward_with_aug(
        self, activations, aug_model, augmented_start=0, augmented_end=None
    ):
        augmented_end = len(self.dec_blocks) if augmented_end is None else augmented_end
        xs = {}
        for i in range(self.len_bias_list):
            b = getattr(self, f"bias_xs_{i}")
            xs[b.shape[2]] = b
        # xs = {a.shape[2]: a for a in self.bias_xs}
        for b, block in enumerate(self.dec_blocks):
            if augmented_end > b >= augmented_start:
                xs = block.forward_with_aug(xs, activations, aug_model, b)
            elif b >= augmented_end:
                xs = block.forward_uncond(xs)
            else:
                xs, _ = block.forward(xs, activations)

        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

    def forward_uncond(self, n, t=None, y=None):
        xs = {}
        for i in range(self.len_bias_list):
            b = getattr(self, f"bias_xs_{i}")
            xs[b.shape[2]] = b.repeat(n, 1, 1, 1)
        # for bias in self.bias_xs:
        #     xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs = block.forward_uncond(xs, temp)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

    def forward_manual_latents(self, n, latents, t=None):
        xs = {}
        for i in range(self.len_bias_list):
            b = getattr(self, f"bias_xs_{i}")
            xs[b.shape[2]] = b.repeat(n, 1, 1, 1)
        # for bias in self.bias_xs:
        #     xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
            xs = block.forward_uncond(xs, t, lvs=lvs)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]


class VAE(HModule):
    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    def load_state_dict(
        self, state_dict: "OrderedDict[str, Tensor]", strict: bool = True
    ):
        state_dict_ = {}
        for k, v in state_dict.items():
            k = k.replace("bias_xs.", "bias_xs_")
            state_dict_[k] = v
        return super().load_state_dict(state_dict_, strict)

    def forward(self, x, x_target, get_recon=False):
        activations = self.encoder.forward(x)
        px_z, stats = self.decoder.forward(activations)
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        ndims = np.prod(x.shape[1:])
        for statdict in stats:
            rate_per_pixel += statdict["kl"].sum(dim=(1, 2, 3))
        rate_per_pixel /= ndims
        elbo = (distortion_per_pixel + rate_per_pixel).mean()
        if get_recon:
            x_recon = self.decoder.out_net.sample(px_z, diffable=True)
            return x_recon, dict(
                elbo=elbo,
                distortion=distortion_per_pixel.mean(),
                rate=rate_per_pixel.mean(),
            )
        return dict(
            elbo=elbo,
            distortion=distortion_per_pixel.mean(),
            rate=rate_per_pixel.mean(),
        )

    def forward_reconstruction(self, x):
        activations = self.encoder.forward(x)
        px_z, _ = self.decoder.forward(activations)
        return self.decoder.out_net.sample(px_z, diffable=True)

    def forward_get_latents(self, x):
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True)
        return stats, activations

    def forward_uncond_samples(self, n_batch, t=None):
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        return self.decoder.out_net.sample(px_z)

    def forward_set_epsilon(self, epsilons, start=0, activations=None):
        px_z, stats = self.decoder.forward_manual_epsilon(activations, epsilons, start)
        return self.decoder.out_net.sample(px_z, diffable=True), stats

    def forward_samples_set_latents(self, n_batch, latents, t=None, diffable=False):
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=t)
        return self.decoder.out_net.sample(px_z, diffable=diffable)

    def forward_with_aug(self, x, aug_model, augmented_start=0, augmented_end=None):
        activations = self.encoder.forward(x)
        px_z = self.decoder.forward_with_aug(
            activations,
            aug_model,
            augmented_start=augmented_start,
            augmented_end=augmented_end,
        )
        return self.decoder.out_net.sample(px_z, diffable=True)


class VAEAugmentor(VAE):
    def forward(self, x):
        return self.forward_reconstruction(x)
