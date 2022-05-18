"""Coupling UMNN transform"""

from nflows.transforms.coupling import CouplingTransform
from nflows.transforms.UMNN.monotonicnormalizer import MonotonicNormalizer

class UMNNCouplingTransform(CouplingTransform):
    """An unconstrained monotonic neural networks coupling layer that transforms the variables.

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
        mask,
        transform_net_create_fn,
        integrand_net_layers=[50, 50, 50],
        cond_size=20,
        nb_steps=20,
        solver="CCParallel",
        apply_unconditional_transform=False
    ):

        if apply_unconditional_transform:
            unconditional_transform = lambda features: MonotonicNormalizer(integrand_net_layers, 0, nb_steps, solver)
        else:
            unconditional_transform = None
        self.cond_size = cond_size
        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

        self.transformer = MonotonicNormalizer(integrand_net_layers, cond_size, nb_steps, solver)

    def _transform_dim_multiplier(self):
        return self.cond_size

    def _coupling_transform_forward(self, inputs, transform_params):
        if len(inputs.shape) == 2:
            z, jac = self.transformer(inputs, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1))
            log_det_jac = jac.log().sum(1)
            return z, log_det_jac
        else:
            B, C, H, W = inputs.shape
            z, jac = self.transformer(inputs.permute(0, 2, 3, 1).reshape(-1, inputs.shape[1]), transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1]))
            log_det_jac = jac.log().reshape(B, -1).sum(1)
            return z.reshape(B, H, W, C).permute(0, 3, 1, 2), log_det_jac

    def _coupling_transform_inverse(self, inputs, transform_params):
        if len(inputs.shape) == 2:
            x = self.transformer.inverse_transform(inputs, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1))
            z, jac = self.transformer(x, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1))
            log_det_jac = -jac.log().sum(1)
            return x, log_det_jac
        else:
            B, C, H, W = inputs.shape
            x = self.transformer.inverse_transform(inputs.permute(0, 2, 3, 1).reshape(-1, inputs.shape[1]), transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1]))
            z, jac = self.transformer(x, transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1]))
            log_det_jac = -jac.log().reshape(B, -1).sum(1)
            return x.reshape(B, H, W, C).permute(0, 3, 1, 2), log_det_jac

