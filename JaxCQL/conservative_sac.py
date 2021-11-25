from collections import OrderedDict
from copy import deepcopy
import functools

from ml_collections import ConfigDict

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import distrax

from .jax_utils import jit_method, next_rng, value_and_multi_grad, mse_loss
from .model import Scalar, update_target_network


class ConservativeSAC(object):

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
        config.use_cql = True
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = 1.0
        config.cql_temp = 1.0
        config.cql_min_q_weight = 5.0

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

        policy_params = self.policy.init(next_rng(), next_rng(), jnp.zeros((10, self.observation_dim)))
        self._train_states['policy'] = TrainState.create(
            params=policy_params,
            tx=optimizer_class(self.config.policy_lr),
            apply_fn=None
        )

        qf1_params = self.qf.init(next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.action_dim)))
        self._train_states['qf1'] = TrainState.create(
            params=qf1_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )
        qf2_params = self.qf.init(next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.action_dim)))
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

        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self._train_states['log_alpha_prime'] = TrainState.create(
                params=self.log_alpha_prime.init(next_rng()),
                tx=optimizer_class(self.config.qf_lr),
                apply_fn=None
            )
            model_keys.append('log_alpha_prime')

        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def train(self, batch):
        self._total_steps += 1
        self._train_states, self._target_qf_params, metrics = self._train_step(
            self._train_states, self._target_qf_params, next_rng(), batch
        )
        return {key: val.item() for key, val in metrics.items()}

    @jit_method
    def _train_step(self, train_states, target_qf_params, rng, batch):

        def loss_fn(train_params, rng):
            observations = batch['observations']
            actions = batch['actions']
            rewards = batch['rewards']
            next_observations = batch['next_observations']
            dones = batch['dones']

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
            new_actions, log_pi = self.policy.apply(train_params['policy'], split_rng, observations)

            if self.config.use_automatic_entropy_tuning:
                alpha_loss = -self.log_alpha.apply(train_params['log_alpha']) * (log_pi + self.config.target_entropy).mean()
                loss_collection['log_alpha'] = alpha_loss
                alpha = jnp.exp(self.log_alpha.apply(train_params['log_alpha'])) * self.config.alpha_multiplier
            else:
                alpha_loss = 0.0
                alpha = self.config.alpha_multiplier

            """ Policy loss """
            q_new_actions = jnp.minimum(
                self.qf.apply(train_params['qf1'], observations, new_actions),
                self.qf.apply(train_params['qf2'], observations, new_actions),
            )
            policy_loss = (alpha*log_pi - q_new_actions).mean()

            loss_collection['policy'] = policy_loss

            """ Q function loss """
            q1_pred = self.qf.apply(train_params['qf1'], observations, actions)
            q2_pred = self.qf.apply(train_params['qf2'], observations, actions)

            rng, split_rng = jax.random.split(rng)
            new_next_actions, next_log_pi = self.policy.apply(
                train_params['policy'], split_rng, next_observations
            )
            target_q_values = jnp.minimum(
                self.qf.apply(target_qf_params['qf1'], next_observations, new_next_actions),
                self.qf.apply(target_qf_params['qf2'], next_observations, new_next_actions),
            )

            if self.config.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            q_target = jax.lax.stop_gradient(
                rewards + (1. - dones) * self.config.discount * target_q_values
            )
            qf1_loss = mse_loss(q1_pred, q_target)
            qf2_loss = mse_loss(q2_pred, q_target)

            ### CQL
            if self.config.use_cql:
                batch_size = actions.shape[0]
                rng, split_rng = jax.random.split(rng)
                cql_random_actions = jax.random.uniform(split_rng, shape=(batch_size, self.config.cql_n_actions, self.action_dim))

                rng, split_rng = jax.random.split(rng)
                cql_current_actions, cql_current_log_pis = self.policy.apply(
                    train_params['policy'], split_rng, observations, repeat=self.config.cql_n_actions
                )
                rng, split_rng = jax.random.split(rng)
                cql_next_actions, cql_next_log_pis = self.policy.apply(
                    train_params['policy'], split_rng, next_observations, repeat=self.config.cql_n_actions
                )

                cql_q1_rand = self.qf.apply(train_params['qf1'], observations, cql_random_actions)
                cql_q2_rand = self.qf.apply(train_params['qf2'], observations, cql_random_actions)
                cql_q1_current_actions = self.qf.apply(train_params['qf1'], observations, cql_current_actions)
                cql_q2_current_actions = self.qf.apply(train_params['qf2'], observations, cql_current_actions)
                cql_q1_next_actions = self.qf.apply(train_params['qf1'], observations, cql_next_actions)
                cql_q2_next_actions = self.qf.apply(train_params['qf2'], observations, cql_next_actions)

                cql_cat_q1 = jnp.concatenate(
                    [cql_q1_rand, jnp.expand_dims(q1_pred, 1), cql_q1_next_actions, cql_q1_current_actions], axis=1
                )
                cql_cat_q2 = jnp.concatenate(
                    [cql_q2_rand, jnp.expand_dims(q2_pred, 1), cql_q2_next_actions, cql_q2_current_actions], axis=1
                )
                cql_std_q1 = jnp.std(cql_cat_q1, axis=1)
                cql_std_q2 = jnp.std(cql_cat_q2, axis=1)

                if self.config.cql_importance_sample:
                    random_density = np.log(0.5 ** self.action_dim)
                    cql_cat_q1 = jnp.concatenate(
                        [cql_q1_rand - random_density,
                         cql_q1_next_actions - cql_next_log_pis,
                         cql_q1_current_actions - cql_current_log_pis],
                        axis=1
                    )
                    cql_cat_q2 = jnp.concatenate(
                        [cql_q2_rand - random_density,
                         cql_q2_next_actions - cql_next_log_pis,
                         cql_q2_current_actions - cql_current_log_pis],
                        axis=1
                    )

                cql_min_qf1_loss = (
                    jax.scipy.special.logsumexp(cql_cat_q1 / self.config.cql_temp, axis=1).mean()
                    * self.config.cql_min_q_weight * self.config.cql_temp
                )
                cql_min_qf2_loss = (
                    jax.scipy.special.logsumexp(cql_cat_q2 / self.config.cql_temp, axis=1).mean()
                    * self.config.cql_min_q_weight * self.config.cql_temp
                )

                """Subtract the log likelihood of data"""
                cql_min_qf1_loss = cql_min_qf1_loss - q1_pred.mean() * self.config.cql_min_q_weight
                cql_min_qf2_loss = cql_min_qf2_loss - q2_pred.mean() * self.config.cql_min_q_weight

                if self.config.cql_lagrange:
                    alpha_prime = jnp.clip(
                        torch.exp(self.log_alpha_prime.apply(train_params['log_alpha_prime'])),
                        a_min=0.0, a_max=1000000.0
                    )
                    cql_min_qf1_loss = alpha_prime * (cql_min_qf1_loss - self.config.cql_target_action_gap)
                    cql_min_qf2_loss = alpha_prime * (cql_min_qf2_loss - self.config.cql_target_action_gap)

                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5

                    loss_collection['log_alpha_prime'] = alpha_prime_loss

                else:
                    alpha_prime_loss = 0.0
                    alpha_prime = 0.0

                qf1_loss = qf1_loss + cql_min_qf1_loss
                qf2_loss = qf2_loss + cql_min_qf2_loss

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

        if self.config.use_cql:
            cql_metrics = dict(
                cql_std_q1=aux_values['cql_std_q1'].mean(),
                cql_std_q2=aux_values['cql_std_q2'].mean(),
                cql_q1_rand=aux_values['cql_q1_rand'].mean(),
                cql_q2_rand=aux_values['cql_q2_rand'].mean(),
                cql_min_qf1_loss=aux_values['cql_min_qf1_loss'].mean(),
                cql_min_qf2_loss=aux_values['cql_min_qf2_loss'].mean(),
                cql_q1_current_actions=aux_values['cql_q1_current_actions'].mean(),
                cql_q2_current_actions=aux_values['cql_q2_current_actions'].mean(),
                cql_q1_next_actions=aux_values['cql_q1_next_actions'].mean(),
                cql_q2_next_actions=aux_values['cql_q2_next_actions'].mean(),
            )
        else:
            cql_metrics = {}

        metrics = dict(
            log_pi=aux_values['log_pi'].mean(),
            policy_loss=aux_values['policy_loss'],
            qf1_loss=aux_values['qf1_loss'],
            qf2_loss=aux_values['qf2_loss'],
            alpha_loss=aux_values['alpha_loss'],
            alpha=aux_values['alpha'],
            average_qf1=aux_values['q1_pred'].mean(),
            average_qf2=aux_values['q2_pred'].mean(),
            average_target_q=aux_values['target_q_values'].mean(),
        )
        metrics.update(cql_metrics)
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
