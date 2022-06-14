"""Autoregressive UMNN transform"""

import torch.nn.functional as F

from ...transforms import made as made_module
from ...transforms.autoregressive import AutoregressiveTransform
from ...transforms.UMNN.monotonicnormalizer import MonotonicNormalizer


class MaskedUMNNAutoregressiveTransform(AutoregressiveTransform):
    """An unconstrained monotonic neural networks autoregressive layer that transforms the variables.

        Reference:
        > A. Wehenkel and G. Louppe, Unconstrained Monotonic Neural Networks, NeurIPS2019.

        ---- Specific arguments ----
        integrand_net_layers: the layers dimension to put in the integrand network.
        cond_size: The embedding size for the conditioning factors.
        nb_steps: The number of integration steps.
        solver: The quadrature algorithm - CC or CCParallel. Both implements Clenshaw-Curtis quadrature with
        Leibniz rule for backward computation. CCParallel pass all the evaluation points (nb_steps) at once, it is faster
        but requires more memory.
        """
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        integrand_net_layers=[50, 50, 50],
        cond_size=20,
        nb_steps=20,
        solver="CCParallel",
    ):
        self.features = features
        self.cond_size = cond_size
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self._epsilon = 1e-3
        super().__init__(made)
        self.transformer = MonotonicNormalizer(integrand_net_layers, cond_size, nb_steps, solver)


    def _output_dim_multiplier(self):
        return self.cond_size

    def _elementwise_forward(self, inputs, autoregressive_params):
        z, jac = self.transformer(inputs, autoregressive_params.reshape(inputs.shape[0], inputs.shape[1], -1))
        log_det_jac = jac.log().sum(1)
        return z, log_det_jac

    def _elementwise_inverse(self, inputs, autoregressive_params):
        x = self.transformer.inverse_transform(inputs, autoregressive_params.reshape(inputs.shape[0], inputs.shape[1], -1))
        z, jac = self.transformer(x, autoregressive_params.reshape(inputs.shape[0], inputs.shape[1], -1))
        log_det_jac = -jac.log().sum(1)
        return x, log_det_jac


