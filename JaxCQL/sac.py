from collections import OrderedDict
from copy import deepcopy
from functools import partial

from ml_collections import ConfigDict

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import distrax

from .jax_utils import (
    next_rng, value_and_multi_grad, mse_loss, JaxRNG, wrap_function_with_rng,
    collect_jax_metrics
)
from .model import Scalar, update_target_network


class SAC(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf):
        self.config = self.get_default_config(config)
        self.policy = policy
        self.qf = qf
        self.observation_dim = policy.observation_dim
        self.action_dim = policy.action_dim

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        policy_params = self.policy.init(
            next_rng(self.policy.rng_keys()),
            jnp.zeros((10, self.observation_dim))
        )
        self._train_states['policy'] = TrainState.create(
            params=policy_params,
            tx=optimizer_class(self.config.policy_lr),
            apply_fn=None
        )

        qf1_params = self.qf.init(
            next_rng(self.qf.rng_keys()),
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim))
        )
        self._train_states['qf1'] = TrainState.create(
            params=qf1_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )
        qf2_params = self.qf.init(
            next_rng(self.qf.rng_keys()),
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim))
        )
        self._train_states['qf2'] = TrainState.create(
            params=qf2_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )
        self._target_qf_params = deepcopy({'qf1': qf1_params, 'qf2': qf2_params})

        model_keys = ['policy', 'qf1', 'qf2']

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self._train_states['log_alpha'] = TrainState.create(
                params=self.log_alpha.init(next_rng()),
                tx=optimizer_class(self.config.policy_lr),
                apply_fn=None
            )
            model_keys.append('log_alpha')

        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def train(self, batch):
        self._total_steps += 1
        self._train_states, self._target_qf_params, metrics = self._train_step(
            self._train_states, self._target_qf_params, next_rng(), batch
        )
        return metrics

    @partial(jax.jit, static_argnames='self')
    def _train_step(self, train_states, target_qf_params, rng, batch):
        rng_generator = JaxRNG(rng)

        def loss_fn(train_params, rng):
            observations = batch['observations']
            actions = batch['actions']
            rewards = batch['rewards']
            next_observations = batch['next_observations']
            dones = batch['dones']

            loss_collection = {}

            @wrap_function_with_rng(rng_generator())
            def forward_policy(rng, *args, **kwargs):
                return self.policy.apply(
                    *args, **kwargs,
                    rngs=JaxRNG(rng)(self.policy.rng_keys())
                )

            @wrap_function_with_rng(rng_generator())
            def forward_qf(rng, *args, **kwargs):
                return self.qf.apply(
                    *args, **kwargs,
                    rngs=JaxRNG(rng)(self.qf.rng_keys())
                )

            new_actions, log_pi = forward_policy(train_params['policy'], observations)

            if self.config.use_automatic_entropy_tuning:
                alpha_loss = -self.log_alpha.apply(train_params['log_alpha']) * (log_pi + self.config.target_entropy).mean()
                loss_collection['log_alpha'] = alpha_loss
                alpha = jnp.exp(self.log_alpha.apply(train_params['log_alpha'])) * self.config.alpha_multiplier
            else:
                alpha_loss = 0.0
                alpha = self.config.alpha_multiplier

            """ Policy loss """
            q_new_actions = jnp.minimum(
                forward_qf(train_params['qf1'], observations, new_actions),
                forward_qf(train_params['qf2'], observations, new_actions),
            )
            policy_loss = (alpha*log_pi - q_new_actions).mean()

            loss_collection['policy'] = policy_loss

            """ Q function loss """
            q1_pred = forward_qf(train_params['qf1'], observations, actions)
            q2_pred = forward_qf(train_params['qf2'], observations, actions)

            new_next_actions, next_log_pi = forward_policy(train_params['policy'], next_observations)
            target_q_values = jnp.minimum(
                forward_qf(target_qf_params['qf1'], next_observations, new_next_actions),
                forward_qf(target_qf_params['qf2'], next_observations, new_next_actions),
            )

            if self.config.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            q_target = jax.lax.stop_gradient(
                rewards + (1. - dones) * self.config.discount * target_q_values
            )
            qf1_loss = mse_loss(q1_pred, q_target)
            qf2_loss = mse_loss(q2_pred, q_target)

            loss_collection['qf1'] = qf1_loss
            loss_collection['qf2'] = qf2_loss

            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }
        new_target_qf_params = {}
        new_target_qf_params['qf1'] = update_target_network(
            new_train_states['qf1'].params, target_qf_params['qf1'],
            self.config.soft_target_update_rate
        )
        new_target_qf_params['qf2'] = update_target_network(
            new_train_states['qf2'].params, target_qf_params['qf2'],
            self.config.soft_target_update_rate
        )

        metrics = collect_jax_metrics(
            aux_values,
            ['log_pi', 'policy_loss', 'qf1_loss', 'qf2_loss', 'alpha_loss',
             'alpha', 'q1_pred', 'q2_pred', 'target_q_values']
        )
        return new_train_states, new_target_qf_params, metrics

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps
