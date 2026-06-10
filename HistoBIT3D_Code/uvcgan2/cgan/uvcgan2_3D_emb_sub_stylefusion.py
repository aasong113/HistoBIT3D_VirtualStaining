# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import itertools
import math
import torch
import os
import matplotlib.pyplot as plt
import numpy as np


import torchvision.transforms.functional as TF
from torchvision.transforms import GaussianBlur, Resize
import torch.nn.functional as F

from uvcgan2.torch.select            import (
    select_optimizer, extract_name_kwargs
)
from uvcgan2.torch.queue             import FastQueue
from uvcgan2.torch.funcs             import prepare_model, update_average_model
from uvcgan2.torch.layers.batch_head import BatchHeadWrapper, get_batch_head
from uvcgan2.base.losses             import GANLoss
from uvcgan2.torch.gradient_penalty  import GradientPenalty
from uvcgan2.models.discriminator    import construct_discriminator
from uvcgan2.models.generator        import construct_generator

from .model_base import ModelBase
from .named_dict import NamedDict
from .funcs import set_two_domain_input, save_image, save_embedding_as_image
from .checkpoint import get_save_path

def construct_consistency_model(consist, device):
    name, kwargs = extract_name_kwargs(consist)

    if name == 'blur':
        return GaussianBlur(**kwargs).to(device)

    if name == 'resize':
        return Resize(**kwargs).to(device)

    raise ValueError(f'Unknown consistency type: {name}')

def queued_forward(batch_head_model, input_image, queue, update_queue = True):
    output, pred_body = batch_head_model.forward(
        input_image, extra_bodies = queue.query(), return_body = True
    )

    if update_queue:
        queue.push(pred_body)

    return output

class UVCGAN2_3D_stylefusion(ModelBase):
    # pylint: disable=too-many-instance-attributes
    # Filename stem used to persist the running-average style token in the same
    # checkpoint directory as the model weights/optimizers.
    STYLE_FUSION_STATE_NAME = "style_fusion_state"

    def _setup_images(self, _config):
        images = [
            'real_a', 'real_b',
            'fake_a', 'fake_b',
            'reco_a', 'reco_b',
            'consist_real_a', 'consist_real_b',
            'consist_fake_a', 'consist_fake_b',
            'real_a_z1', 'real_a_z2',
            'fake_b_z1', 'fake_b_z2',
            'real_a_name', 'real_a_adj_name',
        ]

        if self.is_train and self.lambda_idt > 0:
            images += [ 'idt_a', 'idt_b', ]

        return NamedDict(*images)

    def _construct_batch_head_disc(self, model_config, input_shape):
        disc_body = construct_discriminator(
            model_config, input_shape, self.device
        )

        disc_head = get_batch_head(self.head_config)
        disc_head = prepare_model(disc_head, self.device)

        return BatchHeadWrapper(disc_body, disc_head)

    def _setup_models(self, config):
        models = {}

        shape_a = config.data.datasets[0].shape
        shape_b = config.data.datasets[1].shape

        models['gen_ab'] = construct_generator(
            config.generator, shape_a, shape_b, self.device
        )
        models['gen_ba'] = construct_generator(
            config.generator, shape_b, shape_a, self.device
        )

        if self.avg_momentum is not None:
            models['avg_gen_ab'] = construct_generator(
                config.generator, shape_a, shape_b, self.device
            )
            models['avg_gen_ba'] = construct_generator(
                config.generator, shape_b, shape_a, self.device
            )

            models['avg_gen_ab'].load_state_dict(models['gen_ab'].state_dict())
            models['avg_gen_ba'].load_state_dict(models['gen_ba'].state_dict())

        if self.is_train:
            models['disc_a'] = self._construct_batch_head_disc(
                config.discriminator, config.data.datasets[0].shape
            )
            models['disc_b'] = self._construct_batch_head_disc(
                config.discriminator, config.data.datasets[1].shape
            )

        ## Register ViT Forward Hooks here for embedding loss: 
        embedding_storage = {}

        def make_hook(name):
            def hook(module, input, output):
                embedding_storage[name] = output.detach()
            return hook
        
        # Register hook to bottleneck (ExtendedPixelwiseViT) for both generators
        print("Modules inside gen_ab.net:")
        for name, module in models['gen_ab'].net.named_modules():
            print(name)

        # Get the bottleneck layers AB
        bottleneck_layer_ab = models['gen_ab'].net.modnet.inner_module.inner_module.inner_module.inner_module.encoder.encoder[11].norm2
        bottleneck_layer_ab.register_forward_hook(make_hook("ab"))
        
        bottleneck_layer_ba = models['gen_ba'].net.modnet.inner_module.inner_module.inner_module.inner_module.encoder.encoder[11].norm2
        bottleneck_layer_ba.register_forward_hook(make_hook("ba"))

        self._setup_style_fusion(models)

        return NamedDict(**models)

    def _get_vit_bottleneck(self, gen):
        # `gen.net` is typically a `ModNet` instance; its `.modnet` attribute is
        # the outermost `ModNetBlock` (which does not expose get_bottleneck()).
        # The `ModNet` wrapper provides get_bottleneck() to access the innermost
        # bottleneck module (ExtendedPixelwiseViT).
        if hasattr(gen.net, "get_bottleneck"):
            return gen.net.get_bottleneck()

        # Fallback for unexpected generator wrappers.
        if hasattr(gen, "get_bottleneck"):
            return gen.get_bottleneck()

        raise AttributeError(
            "Could not locate ViT bottleneck; expected `gen.net.get_bottleneck()`."
        )

    def _setup_style_fusion(self, models):
        # --- Style fusion state (running-average B->A style token) ---
        #
        # We treat the ViT bottleneck's *last* "extra token" as a compact
        # per-image style code (see ExtendedPixelwiseViT.forward()).
        #
        # Requested behavior:
        #   - Maintain `style_token_ba` as a *running average* over all B->A
        #     style tokens seen during training (across batches and epochs).
        #   - Save this running-average style token at each checkpoint epoch.
        #   - Load it back at inference time together with model weights, so
        #     A->B can use a fixed, learned-average B-style even without `real_b`.
        #
        # Shape convention:
        #   - running average: (1, feat_dim)
        #   - per-batch tokens: (N, feat_dim) (never stored long-term)
        self.style_token_ba = None  # running-average B->A style token, shape (1, feat_dim)
        self.style_token_ab = None  # cached style token from A->B generator (domain A content; mostly for debugging)
        self.style_delta = None      # deprecated: kept for backwards compatibility; no longer used
        self.style_token_ba_count = 0  # number of samples accumulated into the running mean
        self.style_fusion_enabled = False
        self.style_fusion_handles = {}
        # Gate to control *when* the B->A running average is updated.
        #
        # Important: gen_ba is used in multiple places (cycle reconstruction, idt),
        # but we only want the average to reflect true B-domain inputs (real_b)
        # from the B->A forward direction.
        self.style_ba_update_enabled = False
        # Tracks whether the A->B ViT bottleneck actually performed a style-token
        # modification on the *current* forward pass. We use this to trigger
        # debug image dumps *after* the generator finishes (see forward_dispatch('ab')),
        # because inside the bottleneck hook we do not yet have access to the final
        # generated image tensor `fake_b`.
        self._style_fusion_injected_this_forward = False
        # Tag used to disambiguate which gen_ba forward pass we are observing.
        #
        # gen_ba runs multiple times per training step:
        #   - direction 'ba' : gen_ba(real_b)         -> should be tagged "real_b"
        #   - direction 'ab' : gen_ba(fake_b) (reco_a)-> should be tagged "fake_b"
        #   - direction 'aa' : gen_ba(real_a) (idt)   -> should be untagged/ignored
        #
        # We use this tag inside the bottleneck hook to cache per-iteration
        # style embeddings for the requested style loss.
        self._style_ba_capture_tag = None
        self._style_tokens_ba_real_b = None  # detached target style tokens from gen_ba(real_b), shape (N, feat_dim)
        self._style_tokens_ba_fake_b = None  # style tokens from gen_ba(fake_b) WITH grad, shape (N, feat_dim)

        bottleneck_ba = self._get_vit_bottleneck(models['gen_ba'])
        bottleneck_ab = self._get_vit_bottleneck(models['gen_ab'])
        n_ext = bottleneck_ba.extra_tokens.shape[1]
        feat_dim = bottleneck_ba.extra_tokens.shape[2]

        def capture_style_token_ba(_module, _inputs, output):
            # Capture the *reference style token* from the B->A generator.
            #
            # `output` comes from ExtendedPixelwiseViT.forward():
            #   output[0] = bottleneck feature map
            #   output[1] = flattened extra tokens (N, n_ext * feat_dim)
            #
            # We reshape to (N, n_ext, feat_dim) and take the final extra token
            # as the "style token" we want to inject into A->B.
            mod_flat = output[1]
            mod_tokens = mod_flat.view(mod_flat.shape[0], n_ext, feat_dim)

            # Per-sample style tokens from this forward (N, feat_dim).
            #
            # NOTE: We keep both a detached copy (for running-average updates and
            # as a stable target) and a non-detached tensor (so gradients can
            # flow for the style loss on the fake_b path).
            batch_style = mod_tokens[:, -1, :]

            # --- Per-iteration style token caching (requested style loss) ---
            #
            # Cache the style token for:
            #   - real_b : used as a detached target
            #   - fake_b : kept with grad so the style loss can train gen_ab/gen_ba
            tag = getattr(self, "_style_ba_capture_tag", None)
            if tag == "real_b":
                self._style_tokens_ba_real_b = batch_style.detach()
            elif tag == "fake_b":
                self._style_tokens_ba_fake_b = batch_style

            # During inference/eval we want to keep the checkpointed average fixed.
            #
            # During training, we also only update when explicitly enabled
            # (see `forward_dispatch('ba')`). This prevents contaminating the
            # average with tokens produced from:
            #   - reconstructions (gen_ba(fake_b)) in the A->B cycle,
            #   - identity passes (gen_ba(real_a)) if lambda_idt > 0.
            if (not self.is_train) or (not getattr(self, "style_ba_update_enabled", False)):
                return

            # Streaming mean update over *samples* (not batches).
            #
            # We aggregate a global average across all B->A forwards:
            #   mean_new = (mean_old * count_old + sum(batch_style)) / (count_old + N)
            #
            # This implements the true average style token over training.
            batch_style_detached = batch_style.detach()
            batch_count = int(batch_style_detached.shape[0])
            batch_sum = batch_style_detached.sum(dim=0, keepdim=True)  # (1, feat_dim)

            if self.style_token_ba is None or self.style_token_ba_count <= 0:
                self.style_token_ba = batch_sum / float(batch_count)
                self.style_token_ba_count = batch_count
                return

            total = self.style_token_ba_count + batch_count
            self.style_token_ba = (
                (self.style_token_ba * float(self.style_token_ba_count)) + batch_sum
            ) / float(total)
            self.style_token_ba_count = total

        def capture_style_token_ab(_module, _inputs, output):
            # Capture the A->B style/content token for inspection/debugging.
            # The actual "content token" used for injection is taken from the
            # *current forward pass* inside inject_style_token().
            mod_flat = output[1]
            mod_tokens = mod_flat.view(mod_flat.shape[0], n_ext, feat_dim)
            self.style_token_ab = mod_tokens[:, -1, :].detach()

        def inject_style_token(_module, _inputs, output):
            # Inject style during the A->B forward pass by modifying the last
            # extra token produced by the ViT bottleneck.
            #
            # New behavior (requested):
            #   - content token  = A->B token from the *current* forward pass
            #   - style token    = running-average B->A token accumulated during training
            #   - injection      = AdaIN(content, style) (or simple replacement)
            #
            # We keep `lam` as a continuous knob (cosine-decayed over epochs):
            #   lam=0 => no change
            #   lam=1 => fully use B->A style token
            if not self.style_fusion_enabled or self.style_token_ba is None:
                return output

            result, mod_flat = output
            mod_tokens = mod_flat.view(mod_flat.shape[0], n_ext, feat_dim)

            lam = self._get_style_fusion_lambda()
            # IMPORTANT:
            #   - For 'add' injection, `lam` is the strength of the linear interpolation,
            #     so lam==0 means "do nothing" and we can early-return.
            #   - For 'adain' injection, we want style fusion to remain active even
            #     when lam==0, because AdaIN does not rely on the lambda schedule.
            #     In other words, 'adain' is treated as "always-on" whenever
            #     style fusion is enabled and a style_token_ba exists.
            if self.style_fusion_inject == 'add' and lam == 0.0:
                return output

            # Avoid in-place edits on a view used elsewhere in autograd.
            mod_tokens = mod_tokens.clone()

            # The "content token" is the last extra token coming from the A->B
            # bottleneck on *this* forward pass.
            content_token = mod_tokens[:, -1, :].clone()  # (N, feat_dim)

            # `style_token_ba` is stored as a single average vector (1, feat_dim).
            # Expand it to match the batch so AdaIN can be applied per-sample.
            style_token_ba = self.style_token_ba.to(content_token).expand(
                content_token.shape[0], -1
            )

            if self.style_fusion_inject == 'add':
                # Additive replacement (linear interpolation):
                #   lam=0 => keep original content token
                #   lam=1 => fully replace with the cached B->A style token
                #
                # This uses the *raw* cached style token (style_token_ba) as the
                # target style, per request.
                new_last = content_token + lam * (style_token_ba - content_token)
            elif self.style_fusion_inject == 'adain':
                # AdaIN in token space (requested):
                #   - content = A->B token from the current forward pass
                #   - style   = checkpointed running-average B->A token
                # AdaIN runs regardless of `lam` (see early-return logic above).
                adain_full = self._adain_1d(content_token, style_token_ba)
                new_last = adain_full
            else:
                # Should be prevented by __init__ validation, but keep safe.
                return output

            # Mark that this forward pass actually applied style fusion.
            # We will use this signal later (after gen_ab finishes) to save
            # debug images of the input/output slices at a fixed cadence.
            self._style_fusion_injected_this_forward = True

            new_tokens = torch.cat(
                [mod_tokens[:, :-1, :], new_last.unsqueeze(1)],
                dim=1,
            )
            return (result, new_tokens.reshape(mod_flat.shape[0], -1))

        self.style_fusion_handles["ba_capture"] = (
            bottleneck_ba.register_forward_hook(capture_style_token_ba)
        )
        self.style_fusion_handles["ab_capture"] = (
            bottleneck_ab.register_forward_hook(capture_style_token_ab)
        )
        self.style_fusion_handles["ab_inject"] = (
            bottleneck_ab.register_forward_hook(inject_style_token)
        )

    def _setup_losses(self, config):
        losses = [
            'gen_ab', 'gen_ba', 'cycle_a', 'cycle_b', 'disc_a', 'disc_b',
        ]

        if self.is_train and self.lambda_idt > 0:
            losses += [ 'idt_a', 'idt_b' ]

        if self.is_train and config.gradient_penalty is not None:
            losses += [ 'gp_a', 'gp_b' ]

        if self.consist_model is not None:
            losses += [ 'consist_a', 'consist_b' ]
        
        # This is the subtraction loss that we will start with. 
        if self.is_train and self.lambda_sub_loss > 0: 
            losses += [ 'subtraction_adj']
        
        # This is the embedding loss. 
        if self.is_train and self.lambda_embedding_loss > 0: 
            losses += [ 'embedding_adj']  

        # Style loss computed from gen_ba's ViT bottleneck style token stats.
        if self.is_train and getattr(self, "lambda_style_loss", 0) > 0:
            losses += [ 'style' ]

        return NamedDict(*losses)

    def _setup_optimizers(self, config):
        optimizers = NamedDict('gen', 'disc')

        optimizers.gen = select_optimizer(
            itertools.chain(
                self.models.gen_ab.parameters(),
                self.models.gen_ba.parameters()
            ),
            config.generator.optimizer
        )

        optimizers.disc = select_optimizer(
            itertools.chain(
                self.models.disc_a.parameters(),
                self.models.disc_b.parameters()
            ),
            config.discriminator.optimizer
        )

        return optimizers

    def __init__(
        self, savedir, config, is_train, device, head_config = None,
        lambda_a        = 10.0,
        lambda_b        = 10.0,
        lambda_idt      = 0.5,
        lambda_consist  = 0,
        lambda_subtraction_loss = 0.5, # This is the weight for the subtraction loss between z1 and z2 in the adjacent slice mode.
        lambda_embedding_loss = 0.5, # This is the weight for the embedding loss between adjacent slices.
        lambda_style_loss = 1.0, # Style loss weight comparing gen_ba(real_b) vs gen_ba(fake_b) style-token stats.
        lambda_style_fusion = 1.0, # Scale factor for style token injection; decays with epochs (cosine schedule).
        style_fusion_inject = 'adain', # Injection method for the last ViT style token: 'add' or 'adain'.
        head_queue_size = 3,
        avg_momentum    = None,
        consistency     = None,
        z_spacing      = 1.0, # default. 
        debug_root =  None, # Optional root directory for saving debug images from the subtraction loss.
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        self.lambda_a       = lambda_a
        self.lambda_b       = lambda_b
        self.lambda_idt     = lambda_idt
        self.lambda_consist = lambda_consist
        self.lambda_sub_loss = lambda_subtraction_loss # subtraction Loss
        self.lambda_embedding_loss = lambda_embedding_loss # embedding Losss
        self.lambda_style_loss = lambda_style_loss
        self.lambda_style_fusion_base = lambda_style_fusion
        self.style_fusion_inject = style_fusion_inject
        self.avg_momentum   = avg_momentum
        self.head_config    = head_config or {}
        self.consist_model  = None
        self.z_spacing      = z_spacing
        self.debug_root       = debug_root
        self.current_step =0
        self.total_epochs = getattr(config, "epochs", None)

        if self.style_fusion_inject not in ('add', 'adain'):
            raise ValueError(
                "style_fusion_inject must be one of: 'add', 'adain' "
                f"(got {self.style_fusion_inject!r})"
            )

        print(
            f"Initialized UVCGAN2_3D_embedding_loss with "
            f"lambda_a={lambda_a}, lambda_b={lambda_b}, lambda_idt={lambda_idt}, "
            f"lambda_consist={lambda_consist}, lambda_subtraction_loss={lambda_subtraction_loss}, "
            f"lambda_embedding_loss={lambda_embedding_loss}, lambda_style_loss={lambda_style_loss}, "
            f"lambda_style_fusion={lambda_style_fusion}, style_fusion_inject={style_fusion_inject}, "
            f"avg_momentum={avg_momentum}, z_spacing={z_spacing}"
        )

        # 🔥 Correct handling of debug_root
        if debug_root is not None:
            self.debug_root = debug_root
        elif hasattr(config, "debug_root"):
            self.debug_root = config.debug_root
        else:
            self.debug_root = None

        if self.debug_root is not None:
            os.makedirs(self.debug_root, exist_ok=True)

        if (lambda_consist > 0) and (consistency is not None):
            self.consist_model \
                = construct_consistency_model(consistency, device)

        assert len(config.data.datasets) == 2, \
            "CycleGAN expects a pair of datasets"

        super().__init__(savedir, config, is_train, device)

        self.criterion_gan     = GANLoss(config.loss).to(self.device)
        self.criterion_cycle   = torch.nn.L1Loss()
        self.criterion_idt     = torch.nn.L1Loss()
        self.criterion_consist = torch.nn.L1Loss()
        self.criterion_subtract = torch.nn.L1Loss()  # Loss for the subtraction between adjacent slices in domain A.
        self.criterion_embedding = torch.nn.L1Loss()  # Loss for the embedding consistency between adjacent slices in domain A.
        self.criterion_style = torch.nn.MSELoss()  # L2 (mean-squared) loss for style token statistic matching.

        if self.is_train:
            self.queues = NamedDict(**{
                name : FastQueue(head_queue_size, device = device)
                    for name in [ 'real_a', 'real_b', 'fake_a', 'fake_b' ]
            })

            self.gp = None

            if config.gradient_penalty is not None:
                self.gp = GradientPenalty(**config.gradient_penalty)

    def _get_style_fusion_lambda(self):
        # Cosine decay from lambda_style_fusion_base at epoch 1 to ~0 at epoch total_epochs.
        if self.lambda_style_fusion_base <= 0:
            return 0.0
        if not self.total_epochs or self.total_epochs <= 1:
            return float(self.lambda_style_fusion_base)

        effective_epoch = int(self.epoch) + 1  # model.epoch is the last completed epoch during training
        t = max(0, min(effective_epoch - 1, self.total_epochs - 1))
        T = self.total_epochs - 1
        scale = 0.5 * (1.0 + math.cos(math.pi * (t / T)))
        return float(self.lambda_style_fusion_base) * float(scale)

    @staticmethod
    def _adain_1d(content, style, eps = 1e-6):
        # content/style : (N, C)
        c_mean = content.mean(dim = 1, keepdim = True)
        c_std  = content.var(dim = 1, unbiased = False, keepdim = True).sqrt()

        s_mean = style.mean(dim = 1, keepdim = True)
        s_std  = style.var(dim = 1, unbiased = False, keepdim = True).sqrt()

        normalized = (content - c_mean) / (c_std + eps)
        return normalized * (s_std + eps) + s_mean

    def _set_input(self, inputs, domain):
        set_two_domain_input(self.images, inputs, domain, self.device)

        # Debugging the shape and name
        # print(
        #     "DEBUG: real_a and real_a_adj loaded:",
        #     f"real_a shape = {None if self.images.real_a is None else tuple(self.images.real_a.shape)},",
        #     f"real_a_adj shape = {None if self.images.real_a_adj is None else tuple(self.images.real_a_adj.shape)}"
        # )

        # if hasattr(self.images, "real_a_names"):
        #     print("  real_a_names:", self.images.real_a_names)

        # if hasattr(self.images, "real_a_adj_names"):
        #     print("  real_a_adj_names:", self.images.real_a_adj_names)

        if self.images.real_a is not None:
            if self.consist_model is not None:
                self.images.consist_real_a \
                    = self.consist_model(self.images.real_a)

        if self.images.real_b is not None:
            if self.consist_model is not None:
                self.images.consist_real_b \
                    = self.consist_model(self.images.real_b)

    def cycle_forward_image(self, real, gen_fwd, gen_bkw):
        # pylint: disable=no-self-use

        # (N, C, H, W)
        fake = gen_fwd(real)
        reco = gen_bkw(fake)

        consist_fake = None

        if self.consist_model is not None:
            consist_fake = self.consist_model(fake)

        return (fake, reco, consist_fake)

    # computes the loss in embedding space between adjacent slices. 
    def embedding_loss(self, real_a, real_a_adj, gen_fwd, gen_bkw, step=None):
        """
        Compute motion consistency loss between real and generated slice pairs
        using ViT bottleneck embeddings.
        """

        # --------- 1. Setup embedding storage dict ---------
        if not hasattr(self, "embedding_storage"):
            self.embedding_storage = {}

        def make_hook(name):
            def hook(module, input, output):
                self.embedding_storage[name] = output.detach()
            return hook

        # --------- 2. Register forward hooks (only once) ---------
        if not hasattr(self, "hook_handles"):
            self.hook_handles = {}

            bottleneck_z = gen_fwd.net.modnet.inner_module.inner_module.inner_module.inner_module.encoder.encoder[11].norm2
            bottleneck_fake = gen_bkw.net.modnet.inner_module.inner_module.inner_module.inner_module.encoder.encoder[11].norm2

            self.hook_handles["z"] = bottleneck_z.register_forward_hook(make_hook("z"))
            self.hook_handles["fake"] = bottleneck_fake.register_forward_hook(make_hook("fake"))

        # --------- 3. Forward passes to trigger hooks ---------
        _ = gen_fwd(real_a)
        z1_vit = self.embedding_storage["z"].clone()

        _ = gen_fwd(real_a_adj)
        z2_vit = self.embedding_storage["z"].clone()

        fake_b = gen_fwd(real_a)
        fake_b_adj = gen_fwd(real_a_adj)

        _ = gen_bkw(fake_b)
        fake1_vit = self.embedding_storage["fake"].clone()

        _ = gen_bkw(fake_b_adj)
        fake2_vit = self.embedding_storage["fake"].clone()

        # --------- 4. Compute cosine-based motion maps ---------
        def cosine_motion(a, b):
            a, b = a[:-1], b[:-1]  # remove style token (last) # shape: (N, B, D), style token is the last token. Look at transformer.py to understand. it is concatenated as [patch_tokens, style_token]. We want to remove the style token for motion consistency.
            a = a.squeeze(1)     # (N, D)
            b = b.squeeze(1)

            cos_sim = F.cosine_similarity(a, b, dim=-1)  # (N,)
            motion = 1 - cos_sim                         # higher = more motion
            return motion

        motion_real = cosine_motion(z1_vit, z2_vit)
        motion_fake = cosine_motion(fake1_vit, fake2_vit)
        #print(f"DEBUG: motion_real shape = {motion_real.shape}, motion_fake shape = {motion_fake.shape}")

        # --------- 5. Debug output ---------
        if step is not None and step % 100 == 0:
            def shp(x): return None if x is None else tuple(x.shape)
            print(
                f"[ViT bottleneck @ step {step}] "
                f"z1={shp(z1_vit)}, "
                f"z2={shp(z2_vit)}, "
                f"fake1={shp(fake1_vit)}, "
                f"fake2={shp(fake2_vit)}"
            )

            save_image(real_a[0],          os.path.join(self.debug_root, "real_a_z.png"))
            save_image(real_a_adj[0],          os.path.join(self.debug_root, "real_a_adj_z.png"))
            save_image(fake_b[0],          os.path.join(self.debug_root, "fake_b_z.png"))
            save_image(fake_b_adj[0],          os.path.join(self.debug_root, "fake_b_adj_z.png"))
            save_embedding_as_image(z1_vit, self.debug_root, prefix="embedding_real")
            save_embedding_as_image(fake1_vit, self.debug_root, prefix="embedding_fake")
            save_embedding_as_image(motion_real, self.debug_root, prefix="motion_embedding_real")
            save_embedding_as_image(motion_fake, self.debug_root, prefix="motion_embedding_fake")


        # --------- 6. Motion consistency loss ---------
        return self.criterion_embedding(motion_real, motion_fake)


    # Visualize the ViT Bottle Neck Embeddings for debugging. 
    def save_embedding_as_image(tensor, save_path, prefix="embed", max_channels=3):
        """
        Saves selected channels or a channel-aggregated 2D projection of the embedding tensor.

        Args:
            tensor: (B, C, H, W) torch tensor
            save_path: folder to save the image(s)
            prefix: filename prefix
            max_channels: number of individual channels to save
        """
        os.makedirs(save_path, exist_ok=True)

        tensor = tensor.detach().cpu()

        B, C, H, W = tensor.shape

        for b in range(B):
            # Option 1: save channel-mean projection
            mean_proj = tensor[b].mean(dim=0)
            plt.imsave(os.path.join(save_path, f"{prefix}_b{b}_mean.png"),
                    mean_proj.numpy(), cmap='viridis')

            # Option 2: save first few channels
            for c in range(min(max_channels, C)):
                channel_img = tensor[b, c]
                plt.imsave(os.path.join(save_path, f"{prefix}_b{b}_c{c}.png"),
                        channel_img.numpy(), cmap='gray')



    def subtraction_loss(self, real_a, real_a_adj, gen_fwd, gen_bkw, step=None):
        """
        Compute subtraction consistency loss between adjacent slices using only the first channel.
        Also saves debug subtraction images for inspection.
        """

        # Use only the first channel (C=0) → shape: (B, H, W)
        z1 = real_a[:, 0, :, :]
        z2 = real_a_adj[:, 0, :, :]
        #print("DEBUG: z-spacing = :", self.z_spacing)
        subtract_real = torch.abs(z1 - z2) / self.z_spacing  # shape: (B, H, W)

        # Generate fake images
        fake_b = gen_fwd(real_a)
        fake_b_adj = gen_fwd(real_a_adj)
        recon_a = gen_bkw(fake_b)
        recon_a_adj = gen_bkw(fake_b_adj)

        # Use only first channel of generated images
        fake_z1 = recon_a[:, 0, :, :]
        fake_z2 = recon_a_adj[:, 0, :, :]

        subtract_fake = torch.abs(fake_z1 - fake_z2) /self.z_spacing

        #print(fake_b.shape, z1.shape, z2.shape, fake_z1.shape, fake_z2.shape, subtract_real.shape, subtract_fake.shape)
        if step % 100 == 0: 
            # Save subtraction images for debugging (just first sample)
            # Debug images. In practice, you might want to save these less frequently or only a few samples.
            os.makedirs(self.debug_root, exist_ok=True)
            # Only save first sample in batch

            z1_filename = self.images.real_a_names[0]  # string: "img=123_P=0.tif"
            z2_filename = self.images.real_a_adj_names[0]
            print(f"[Debug] z1 name: {z1_filename}, z2 name: {z2_filename}")
            save_image(fake_b[0],          os.path.join(self.debug_root, "fake_b_z.png"))
            save_image(z1[0],          os.path.join(self.debug_root, "z1_real.png"))
            save_image(z2[0],          os.path.join(self.debug_root, "z2_real.png"))
            save_image(fake_z1[0],     os.path.join(self.debug_root, "z1_fake.png"))
            save_image(fake_z2[0],     os.path.join(self.debug_root, "z2_fake.png"))
            save_image(subtract_real[0], os.path.join(self.debug_root, "subtraction_real.png"))
            save_image(subtract_fake[0], os.path.join(self.debug_root, "subtraction_fake.png"))

        # Compute L1 loss between subtraction maps
        return self.criterion_subtract(subtract_real, subtract_fake)


    def idt_forward_image(self, real, gen):
        # pylint: disable=no-self-use

        # (N, C, H, W)
        idt = gen(real)
        return idt
    
    # Function to save the forward images for debugging.
    def save_forward_image(self, real_a, real_a_adj, gen_ab, step=None):
        # pylint: disable=no-self-use

        # (N, C, H, W)
        fake_b = gen_ab(real_a)
        fake_b_adj = gen_ab(real_a_adj)
        if step % 100 == 0: 
            # Save subtraction images for debugging (just first sample)
            # Debug images. In practice, you might want to save these less frequently or only a few samples.
            os.makedirs(self.debug_root, exist_ok=True)
            save_image(real_a[0],          os.path.join(self.debug_root, "real_a.png"))
            save_image(real_a_adj[0],      os.path.join(self.debug_root, "real_a_adj.png"))
            save_image(fake_b[0],          os.path.join(self.debug_root, "fake_b.png"))
            save_image(fake_b_adj[0],      os.path.join(self.debug_root, "fake_b_adj.png"))

        return None

    def forward_dispatch(self, direction):
        if direction == 'ab':
            # Ensure we do NOT update the running B->A style mean when gen_ba is
            # used as the backward cycle (gen_ba(fake_b)).
            self.style_ba_update_enabled = False
            # Only enable style injection for the "real_a -> fake_b" A->B path.
            # This prevents unintended injection when gen_ab is used as the
            # backward cycle in the opposite direction.
            self.style_fusion_enabled = self.style_token_ba is not None
            # Style fusion is enabled whenever we have a persisted/learned
            # running-average B->A style token.
            #
            # This is intentionally *independent* of whether `real_b` is present
            # in the current batch:
            #   - During training, `style_token_ba` is updated whenever gen_ba
            #     runs (direction 'ba').
            #   - During inference, `style_token_ba` is loaded from checkpoint,
            #     so A->B can use the same fixed style without needing paired B.
            self.style_fusion_enabled = self.style_token_ba is not None
            # Reset per-forward flag; the ViT bottleneck hook will flip this to
            # True only if it actually applied a style-token modification.
            self._style_fusion_injected_this_forward = False

            # Tag the upcoming gen_ba(fake_b) reconstruction pass so the gen_ba
            # bottleneck hook can cache per-iteration style tokens for the style loss.
            self._style_ba_capture_tag = "fake_b"

            (
                self.images.fake_b, self.images.reco_a,
                self.images.consist_fake_b
            ) = self.cycle_forward_image(
                self.images.real_a, self.models.gen_ab, self.models.gen_ba
            )
            # Clear tag after gen_ba(fake_b) has executed inside cycle_forward_image().
            self._style_ba_capture_tag = None

            # Save style-fusion debug images (every 100 training steps) *after*
            # the full A->B generator forward completes.
            #
            # Why here (and not inside inject_style_token()):
            #   - The bottleneck hook only sees intermediate features/tokens.
            #   - We want to save the final generated images (fake_b), which are
            #     only available after gen_ab returns.
            #
            # What we save:
            #   - realA_z        : `real_a` (z slice)
            #   - realA_z_adj    : `real_a_adj` (z+1 slice), if present
            #   - fakeB_z        : `fake_b` generated from realA_z
            #   - fakeB_z_adj    : gen_ab(realA_z_adj), generated on-demand for debugging
            if (
                self.is_train
                and self.debug_root is not None
                and (self.current_step % 1000 == 0)
                and self._style_fusion_injected_this_forward
            ):
                os.makedirs(self.debug_root, exist_ok=True)

                def _safe_name(name):
                    # Make filenames robust to characters like '/' or '=' that can
                    # appear in dataset-provided image IDs.
                    # Also strip known image suffixes so we don't end up with
                    # filenames like "..._img=0_P=1.tif_realA_z.png".
                    base = os.path.basename(str(name))
                    while True:
                        root, ext = os.path.splitext(base)
                        if ext.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"} and root:
                            base = root
                            continue
                        break
                    return base.replace(os.sep, "_").replace("/", "_").replace("\\", "_").replace("=", "-")

                step = int(self.current_step)
                # Always include the global training step in filenames to avoid
                # overwriting when the same patch appears multiple times.
                step_tag = f"step{step:06d}"
                z_name = (
                    _safe_name(self.images.real_a_names[0])
                    if hasattr(self.images, "real_a_names") and self.images.real_a_names
                    else "realA"
                )
                z_adj_name = (
                    _safe_name(self.images.real_a_adj_names[0])
                    if hasattr(self.images, "real_a_adj_names") and self.images.real_a_adj_names
                    else None
                )

                # Save z slice input/output (always available).
                save_image(
                    self.images.real_a[0],
                    os.path.join(self.debug_root, f"stylefusion_{step_tag}_{z_name}_realA_z.png"),
                )
                save_image(
                    self.images.fake_b[0],
                    os.path.join(self.debug_root, f"stylefusion_{step_tag}_{z_name}_fakeB_z.png"),
                )

                # Save z+1 slice input/output if the dataset provides it (adjacent-z mode).
                if hasattr(self.images, "real_a_adj") and self.images.real_a_adj is not None:
                    tag = z_adj_name or f"{z_name}_adj"
                    save_image(
                        self.images.real_a_adj[0],
                        os.path.join(self.debug_root, f"stylefusion_{step_tag}_{tag}_realA_z_adj.png"),
                    )
                    # Run A->B on the adjacent slice to visualize the z+1 output.
                    # This forward is in no_grad() so it does not affect training.
                    with torch.no_grad():
                        fake_b_adj = self.models.gen_ab(self.images.real_a_adj)
                    save_image(
                        fake_b_adj[0],
                        os.path.join(self.debug_root, f"stylefusion_{step_tag}_{tag}_fakeB_z_adj.png"),
                    )

        elif direction == 'ba':
            # Do NOT inject style into gen_ab when it is used as the backward
            # cycle (reconstruction) for the B->A direction.
            self.style_fusion_enabled = False
            # Enable updating the running-average B->A style token for *real_b*
            # passes only. The forward hook on the B->A bottleneck will read
            # this flag and update the streaming mean when True.
            self.style_ba_update_enabled = True
            # Tag the gen_ba(real_b) forward so the bottleneck hook caches the
            # "real B" style token for the style loss target.
            self._style_ba_capture_tag = "real_b"

            (
                self.images.fake_a, self.images.reco_b,
                self.images.consist_fake_a
            ) = self.cycle_forward_image(
                self.images.real_b, self.models.gen_ba, self.models.gen_ab
            )
            # Clear tag after gen_ba(real_b) has executed inside cycle_forward_image().
            self._style_ba_capture_tag = None
            # Disable immediately after the B->A forward so other gen_ba calls
            # (e.g., idt_a or cycle reconstruction) don't affect the average.
            self.style_ba_update_enabled = False

        elif direction == 'aa':
            self.style_fusion_enabled = False
            # Identity pass through gen_ba(real_a) should NOT update the B-style mean.
            self.style_ba_update_enabled = False
            self.images.idt_a = \
                self.idt_forward_image(self.images.real_a, self.models.gen_ba)

        elif direction == 'bb':
            self.style_fusion_enabled = False
            self.style_ba_update_enabled = False
            self.images.idt_b = \
                self.idt_forward_image(self.images.real_b, self.models.gen_ab)

        elif direction == 'avg-ab':
            (
                self.images.fake_b, self.images.reco_a,
                self.images.consist_fake_b
            ) = self.cycle_forward_image(
                self.images.real_a,
                self.models.avg_gen_ab, self.models.avg_gen_ba
            )

        elif direction == 'avg-ba':
            (
                self.images.fake_a, self.images.reco_b,
                self.images.consist_fake_a
            ) = self.cycle_forward_image(
                self.images.real_b,
                self.models.avg_gen_ba, self.models.avg_gen_ab
            )

        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")

    def _save_model_state(self, epoch):
        # Persist the running-average style token to disk alongside checkpoints.
        #
        # Why:
        #   - The running mean is NOT part of any torch.nn.Module parameters,
        #     so it would otherwise be lost at save/load.
        #   - Inference needs a stable "B style" even without access to real_b.
        #
        # What we save:
        #   - style_token_ba       : averaged style vector (1, feat_dim) on CPU
        #   - style_token_ba_count : number of samples contributing to the mean
        state = {
            "style_token_ba": None if self.style_token_ba is None else self.style_token_ba.detach().cpu(),
            "style_token_ba_count": int(getattr(self, "style_token_ba_count", 0)),
        }

        save_path = get_save_path(
            self.savedir, self.STYLE_FUSION_STATE_NAME, epoch, mkdir=True
        )
        torch.save(state, save_path)

    def _load_model_state(self, epoch):
        # Restore the running-average style token from disk.
        #
        # Backward compatibility:
        #   - Older checkpoints may not have this file; in that case, style
        #     fusion simply stays disabled until training accumulates a mean.
        load_path = get_save_path(
            self.savedir, self.STYLE_FUSION_STATE_NAME, epoch, mkdir=False
        )
        if not os.path.exists(load_path):
            return

        state = torch.load(load_path, map_location=self.device)
        self.style_token_ba_count = int(state.get("style_token_ba_count", 0))

        token = state.get("style_token_ba", None)
        if token is None:
            self.style_token_ba = None
            return

        # Keep the (1, feat_dim) convention on the active device.
        token = token.to(self.device)
        if token.ndim == 1:
            token = token.unsqueeze(0)
        self.style_token_ba = token

    def forward(self):
        if self.images.real_a is not None:
            if self.avg_momentum is not None:
                self.forward_dispatch(direction = 'avg-ab')
            else:
                self.forward_dispatch(direction = 'ab')

        if self.images.real_b is not None:
            if self.avg_momentum is not None:
                self.forward_dispatch(direction = 'avg-ba')
            else:
                self.forward_dispatch(direction = 'ba')


    def eval_consist_loss(
        self, consist_real_0, consist_fake_1, lambda_cycle_0
    ):
        return lambda_cycle_0 * self.lambda_consist * self.criterion_consist(
            consist_fake_1, consist_real_0
        )

    def eval_loss_of_cycle_forward(
        self, disc_1, real_0, fake_1, reco_0, fake_queue_1, lambda_cycle_0
    ):
        # pylint: disable=too-many-arguments
        # NOTE: Queue is updated in discriminator backprop
        disc_pred_fake_1 = queued_forward(
            disc_1, fake_1, fake_queue_1, update_queue = False
        )

        loss_gen   = self.criterion_gan(disc_pred_fake_1, True)
        loss_cycle = lambda_cycle_0 * self.criterion_cycle(reco_0, real_0)

        loss = loss_gen + loss_cycle

        return (loss_gen, loss_cycle, loss)

    def eval_loss_of_idt_forward(self, real_0, idt_0, lambda_cycle_0):
        loss_idt = (
              lambda_cycle_0
            * self.lambda_idt
            * self.criterion_idt(idt_0, real_0)
        )

        loss = loss_idt

        return (loss_idt, loss)

    def eval_style_loss_ba_vit_token_stats(self):
        """
        Style loss based on gen_ba's ViT bottleneck *style token* statistics.

        We compare, per training iteration:
          - "style source" : style token from gen_ba(real_b)
          - "style target" : style token from gen_ba(fake_b)  (fake_b = gen_ab(real_a))

        The loss matches the *mean* and *standard deviation* (computed over the
        feature dimension of the token embedding) using an L2/MSE penalty:

          L_s = || mu(fake) - mu(real) ||^2 + || sigma(fake) - sigma(real) ||^2

        Notes:
          - This uses the *per-iteration* cached tokens (NOT the running-average
            token used for style injection).
          - The "real_b" token is detached so it acts as a fixed style target
            for this step; the "fake_b" token keeps gradients.
        """
        real_tok = getattr(self, "_style_tokens_ba_real_b", None)
        fake_tok = getattr(self, "_style_tokens_ba_fake_b", None)
        if real_tok is None or fake_tok is None:
            return None

        # Handle rare batch-size mismatches safely.
        if real_tok.shape[0] != fake_tok.shape[0]:
            return None

        # Compute per-sample mean/std over feature dimension.
        # Tokens are shaped (N, feat_dim), so mu/sigma are (N,).
        mu_real = real_tok.mean(dim=1)
        mu_fake = fake_tok.mean(dim=1)

        # Use population std (unbiased=False) for stability on small dims.
        std_real = real_tok.var(dim=1, unbiased=False).sqrt()
        std_fake = fake_tok.var(dim=1, unbiased=False).sqrt()

        loss_mu = self.criterion_style(mu_fake, mu_real)
        loss_std = self.criterion_style(std_fake, std_real)
        return loss_mu + loss_std

    def backward_gen(self, direction):
        if direction == 'ab':
            (self.losses.gen_ab, self.losses.cycle_a, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc_b,
                    self.images.real_a, self.images.fake_b, self.images.reco_a,
                    self.queues.fake_b, self.lambda_a
                )

            if self.consist_model is not None:
                self.losses.consist_a = self.eval_consist_loss(
                    self.images.consist_real_a, self.images.consist_fake_b,
                    self.lambda_a
                )

                loss += self.losses.consist_a

             # ✅ Subtraction loss (adjacent z slices)
            if self.lambda_sub_loss > 0 and hasattr(self.images, "real_a_adj"):
                self.losses.subtraction_adj = self.subtraction_loss(
                    self.images.real_a,
                    self.images.real_a_adj,
                    self.models.gen_ab,
                    self.models.gen_ba,
                    step=self.current_step
                )

                loss += self.lambda_sub_loss * self.losses.subtraction_adj
            
            # Embedding Loss (adjacent z slices)
            if self.lambda_embedding_loss > 0 and hasattr(self.images, "real_a_adj"):
                self.losses.embedding_adj = self.embedding_loss(
                    self.images.real_a,
                    self.images.real_a_adj,
                    self.models.gen_ab,
                    self.models.gen_ba,
                    step=self.current_step
                )

                loss += self.lambda_embedding_loss * self.losses.embedding_adj
            
            # if running original UVCGAN2 without adjacent slice losses.  
            if self.lambda_embedding_loss == 0 and self.lambda_sub_loss == 0 and hasattr(self.images, "real_a_adj"):
                self.save_forward_image(self.images.real_a, self.images.real_a_adj, self.models.gen_ab, step=self.current_step)

            # --- Style loss (gen_ba ViT style-token stats) ---
            #
            # This style loss is independent of style *injection*; it can be used
            # even when lambda_style_fusion == 0.
            #
            # It relies on the gen_ba bottleneck hook having cached:
            #   - _style_tokens_ba_real_b from direction 'ba' (gen_ba(real_b))
            #   - _style_tokens_ba_fake_b from direction 'ab' (gen_ba(fake_b) inside reconstruction)
            if getattr(self, "lambda_style_loss", 0) > 0:
                style_loss = self.eval_style_loss_ba_vit_token_stats()
                if style_loss is not None:
                    self.losses.style = style_loss
                    loss += float(self.lambda_style_loss) * self.losses.style

            # Clear cached fake token to avoid holding a computation graph longer
            # than necessary (prevents memory growth over training).
            self._style_tokens_ba_fake_b = None


        elif direction == 'ba':
            (self.losses.gen_ba, self.losses.cycle_b, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc_a,
                    self.images.real_b, self.images.fake_a, self.images.reco_b,
                    self.queues.fake_a, self.lambda_b
                )

            if self.consist_model is not None:
                self.losses.consist_b = self.eval_consist_loss(
                    self.images.consist_real_b, self.images.consist_fake_a,
                    self.lambda_b
                )

                loss += self.losses.consist_b

        elif direction == 'aa':
            (self.losses.idt_a, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_a, self.images.idt_a, self.lambda_a
                )

        elif direction == 'bb':
            (self.losses.idt_b, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_b, self.images.idt_b, self.lambda_b
                )
        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")


        loss.backward()

    def backward_discriminator_base(
        self, model, real, fake, queue_real, queue_fake
    ):
        # pylint: disable=too-many-arguments
        loss_gp = None

        if self.gp is not None:
            loss_gp = self.gp(
                model, fake, real,
                model_kwargs_fake = { 'extra_bodies' : queue_fake.query() },
                model_kwargs_real = { 'extra_bodies' : queue_real.query() },
            )
            loss_gp.backward()

        # NOTE:
        # `FastQueue.push()` updates the underlying queue tensor in-place. If we
        # push into the queue before backprop, autograd can error with:
        #   "variable needed for gradient computation has been modified by an inplace operation"
        # So we defer queue updates until *after* `loss.backward()`.
        pred_real, pred_body_real = model.forward(
            real, extra_bodies=queue_real.query(), return_body=True
        )
        loss_real = self.criterion_gan(pred_real, True)

        pred_fake, pred_body_fake = model.forward(
            fake, extra_bodies=queue_fake.query(), return_body=True
        )
        loss_fake = self.criterion_gan(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5
        loss.backward()

        queue_real.push(pred_body_real)
        queue_fake.push(pred_body_fake)

        return (loss_gp, loss)

    def backward_discriminators(self):
        fake_a = self.images.fake_a.detach()
        fake_b = self.images.fake_b.detach()

        loss_gp_b, self.losses.disc_b \
            = self.backward_discriminator_base(
                self.models.disc_b, self.images.real_b, fake_b,
                self.queues.real_b, self.queues.fake_b
            )

        if loss_gp_b is not None:
            self.losses.gp_b = loss_gp_b

        loss_gp_a, self.losses.disc_a = \
            self.backward_discriminator_base(
                self.models.disc_a, self.images.real_a, fake_a,
                self.queues.real_a, self.queues.fake_a
            )

        if loss_gp_a is not None:
            self.losses.gp_a = loss_gp_a

    def optimization_step_gen(self):
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], False)
        self.optimizers.gen.zero_grad(set_to_none = True)

        # Run B->A first so the running-average B-style token is updated from
        # the current `real_b` batch before it is used for A->B style injection.
        dir_list = [ 'ba', 'ab' ]
        if self.lambda_idt > 0:
            dir_list += [ 'aa', 'bb' ]

        for direction in dir_list:
            self.forward_dispatch(direction)
            self.backward_gen(direction)

        self.optimizers.gen.step()

    def optimization_step_disc(self):
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], True)
        self.optimizers.disc.zero_grad(set_to_none = True)

        self.backward_discriminators()

        self.optimizers.disc.step()

    def _accumulate_averages(self):
        update_average_model(
            self.models.avg_gen_ab, self.models.gen_ab, self.avg_momentum
        )
        update_average_model(
            self.models.avg_gen_ba, self.models.gen_ba, self.avg_momentum
        )

    def optimization_step(self):
        self.optimization_step_gen()
        self.optimization_step_disc()

        if self.avg_momentum is not None:
            self._accumulate_averages()
        
        self.current_step += 1
