import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder


class A2CSAPGPrivBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = kwargs.pop("num_seqs", 1)
            self.net_type = kwargs.pop("type", "simple")

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            if len(input_shape) != 1:
                raise ValueError("actor_critic_sapg_priv only supports flat 1D observations")

            self.original_input_shape = input_shape[0]
            self.expl_feature_dim = 0
            self.param_ids = None
            self.pid_idx = None
            self.extra_params = None
            self.critic_policy_contact_dim = int(self.sapg_priv_cfg.get("critic_policy_contact_dim", 0))

            if self.critic_policy_contact_dim > 0 and not self.separate:
                raise ValueError(
                    "critic_policy_contact_dim > 0 requires network.separate=True, "
                    "otherwise actor and critic are forced to share the same fused input."
                )

            if self.net_type == "extra_param":
                self.param_ids = kwargs["coef_ids"]
                self.pid_idx = kwargs["coef_id_idx"]
                self.expl_feature_dim = self.expl_embedding_dim
                self.extra_params = nn.Parameter(
                    torch.randn((len(self.param_ids), self.expl_feature_dim), dtype=torch.float32, requires_grad=True),
                    requires_grad=True,
                )
                self.base_obs_dim = self.pid_idx - self.critic_policy_contact_dim
            else:
                self.base_obs_dim = self.policy_obs_dim + self.privileged_obs_dim
                self.expl_feature_dim = self.original_input_shape - self.base_obs_dim - self.critic_policy_contact_dim

            if self.policy_obs_dim + self.privileged_obs_dim != self.base_obs_dim:
                raise ValueError(
                    "policy_obs_dim + privileged_obs_dim must equal the base observation size before SAPG exploration features"
                )

            if self.expl_feature_dim < 0:
                raise ValueError("expl_feature_dim became negative; check your observation split config")

            self.priv_encoder = self._build_privileged_encoder(self.privileged_obs_dim)
            if self.separate:
                self.critic_priv_encoder = self._build_privileged_encoder(self.privileged_obs_dim)
            else:
                self.critic_priv_encoder = None

            fused_actor_input = self.policy_obs_dim + self.privileged_latent_dim + self.expl_feature_dim
            fused_critic_input = fused_actor_input + self.critic_policy_contact_dim

            if len(self.units) == 0:
                actor_out_size = fused_actor_input
                critic_out_size = fused_critic_input
            else:
                actor_out_size = self.units[-1]
                critic_out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    a_rnn_in_size = actor_out_size
                    c_rnn_in_size = critic_out_size
                    actor_out_size = self.rnn_units
                    critic_out_size = self.rnn_units
                    if self.rnn_concat_input:
                        a_rnn_in_size += fused_actor_input
                        c_rnn_in_size += fused_critic_input
                else:
                    a_rnn_in_size = fused_actor_input
                    c_rnn_in_size = fused_critic_input
                    fused_actor_input = self.rnn_units
                    fused_critic_input = self.rnn_units

                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, a_rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, c_rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, a_rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            actor_mlp_args = {
                "input_size": fused_actor_input,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": torch.nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }
            self.actor_mlp = self._build_mlp(**actor_mlp_args)
            if self.separate:
                critic_mlp_args = dict(actor_mlp_args)
                critic_mlp_args["input_size"] = fused_critic_input
                self.critic_mlp = self._build_mlp(**critic_mlp_args)
            else:
                self.critic_mlp = self.actor_mlp

            self.value = self._build_value_layer(critic_out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(actor_out_size, actions_num)
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(actor_out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(actor_out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config["mu_activation"])
                mu_init = self.init_factory.create(**self.space_config["mu_init"])
                self.sigma_act = self.activations_factory.create(self.space_config["sigma_activation"])
                sigma_init = self.init_factory.create(**self.space_config["sigma_init"])

                if self.fixed_sigma == "fixed":
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                elif self.fixed_sigma == "coef_cond":
                    self.sigma_ids = kwargs["coef_ids"]
                    self.sigma_id_idx = kwargs["coef_id_idx"]
                    self.sigma = nn.Parameter(
                        torch.zeros(len(self.sigma_ids), actions_num, requires_grad=True, dtype=torch.float32),
                        requires_grad=True,
                    )
                else:
                    self.sigma = torch.nn.Linear(actor_out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    mlp_init(module.weight)
                    if getattr(module, "bias", None) is not None:
                        torch.nn.init.zeros_(module.bias)

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma != "obs_cond":
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            self.last_privileged_latent = None
            self.actor_encoder_obs_override = None

        def _build_privileged_encoder(self, input_size):
            if input_size == 0:
                self.privileged_latent_dim = 0
                return nn.Identity()

            encoder_cfg = self.sapg_priv_cfg.get("encoder", {})
            units = encoder_cfg.get("units", [])
            if not units:
                self.privileged_latent_dim = input_size
                return nn.Identity()

            self.privileged_latent_dim = units[-1]
            return self._build_mlp(
                input_size=input_size,
                units=units,
                activation=encoder_cfg.get("activation", self.activation),
                dense_func=torch.nn.Linear,
                norm_func_name=encoder_cfg.get("normalization", self.normalization),
                d2rl=encoder_cfg.get("d2rl", False),
                norm_only_first_layer=encoder_cfg.get("norm_only_first_layer", False),
            )

        def _extract_parts(self, raw_obs):
            base_obs = raw_obs[:, :self.base_obs_dim]
            policy_obs = base_obs[:, :self.policy_obs_dim]
            privileged_obs = base_obs[:, self.policy_obs_dim:self.policy_obs_dim + self.privileged_obs_dim]
            critic_policy_contact = raw_obs[:, self.base_obs_dim:self.base_obs_dim + self.critic_policy_contact_dim]

            if self.net_type == "extra_param":
                idxs = (raw_obs[:, self.pid_idx].reshape(-1, 1) == self.param_ids).float().argmax(dim=1)
                expl_features = self.extra_params[idxs]
            else:
                expl_features = raw_obs[:, self.base_obs_dim + self.critic_policy_contact_dim:]
            return policy_obs, privileged_obs, critic_policy_contact, expl_features

        def _get_actor_encoder_obs(self, privileged_obs):
            if self.actor_encoder_obs_override is None:
                return privileged_obs
            if self.actor_encoder_obs_override.shape[0] != privileged_obs.shape[0]:
                raise ValueError(
                    "actor_encoder_obs_override batch dimension must match the current observation batch"
                )
            return self.actor_encoder_obs_override

        def _fuse_actor_inputs(self, raw_obs):
            policy_obs, privileged_obs, _, expl_features = self._extract_parts(raw_obs)
            encoder_obs = self._get_actor_encoder_obs(privileged_obs)
            latent = self.priv_encoder(encoder_obs) if encoder_obs.shape[1] > 0 else encoder_obs
            self.last_privileged_latent = latent
            parts = [policy_obs]
            if latent.shape[1] > 0:
                parts.append(latent)
            if expl_features.shape[1] > 0:
                parts.append(expl_features)
            return torch.cat(parts, dim=1)

        def _fuse_critic_inputs(self, raw_obs):
            policy_obs, privileged_obs, critic_policy_contact, expl_features = self._extract_parts(raw_obs)
            if self.separate:
                latent = self.critic_priv_encoder(privileged_obs) if self.privileged_obs_dim > 0 else privileged_obs
            else:
                latent = self.last_privileged_latent
            parts = [policy_obs]
            if critic_policy_contact.shape[1] > 0:
                parts.append(critic_policy_contact)
            if latent.shape[1] > 0:
                parts.append(latent)
            if expl_features.shape[1] > 0:
                parts.append(expl_features)
            return torch.cat(parts, dim=1)

        def forward(self, obs_dict):
            raw_obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)
            dones = obs_dict.get("dones", None)
            bptt_len = obs_dict.get("bptt_len", 0)
            seq_length = obs_dict.get("seq_length", 1)

            if self.separate:
                a_out = self._fuse_actor_inputs(raw_obs)
                c_out = self._fuse_critic_inputs(raw_obs)

                if self.has_rnn:
                    if not self.is_rnn_before_mlp:
                        a_out_in = a_out
                        c_out_in = c_out
                        a_out = self.actor_mlp(a_out_in)
                        c_out = self.critic_mlp(c_out_in)
                        if self.rnn_concat_input:
                            a_out = torch.cat([a_out, a_out_in], dim=1)
                            c_out = torch.cat([c_out, c_out_in], dim=1)

                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                    c_out = c_out.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)

                    if len(states) == 2:
                        a_states = states[0]
                        c_states = states[1]
                    else:
                        a_states = states[:2]
                        c_states = states[2:]

                    a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                    c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)
                    a_out = a_out.transpose(0, 1).contiguous().reshape(-1, a_out.size(-1))
                    c_out = c_out.transpose(0, 1).contiguous().reshape(-1, c_out.size(-1))

                    if self.rnn_ln:
                        a_out = self.a_layer_norm(a_out)
                        c_out = self.c_layer_norm(c_out)

                    if type(a_states) is not tuple:
                        a_states = (a_states,)
                        c_states = (c_states,)
                    states = a_states + c_states

                    if self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)
                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)

                value = self.value_act(self.value(c_out))

                if self.is_discrete:
                    return self.logits(a_out), value, states
                if self.is_multi_discrete:
                    return [logit(a_out) for logit in self.logits], value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.fixed_sigma == "fixed":
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    elif self.fixed_sigma == "coef_cond":
                        idxs = (raw_obs[:, self.sigma_id_idx].reshape(-1, 1) == self.sigma_ids).float().argmax(dim=1)
                        sigma = self.sigma_act(self.sigma[idxs])
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                    return mu, sigma, value, states
            else:
                out = self._fuse_actor_inputs(raw_obs)
                if self.has_rnn:
                    out_in = out
                    if not self.is_rnn_before_mlp:
                        out = self.actor_mlp(out)
                        if self.rnn_concat_input:
                            out = torch.cat([out, out_in], dim=1)

                    batch_size = out.size()[0]
                    num_seqs = batch_size // seq_length
                    out = out.reshape(num_seqs, seq_length, -1)

                    if len(states) == 1:
                        states = states[0]

                    out = out.transpose(0, 1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                    out, states = self.rnn(out, states, dones, bptt_len)
                    out = out.transpose(0, 1).contiguous().reshape(-1, out.size(-1))

                    if self.rnn_ln:
                        out = self.layer_norm(out)
                    if self.is_rnn_before_mlp:
                        out = self.actor_mlp(out)
                    if type(states) is not tuple:
                        states = (states,)
                else:
                    out = self.actor_mlp(out)

                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states
                if self.is_discrete:
                    return self.logits(out), value, states
                if self.is_multi_discrete:
                    return [logit(out) for logit in self.logits], value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.fixed_sigma == "fixed":
                        sigma = self.sigma_act(self.sigma)
                    elif self.fixed_sigma == "coef_cond":
                        idxs = (raw_obs[:, self.sigma_id_idx].reshape(-1, 1) == self.sigma_ids).float().argmax(dim=1)
                        sigma = self.sigma_act(self.sigma[idxs])
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu * 0 + sigma, value, states

        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            num_layers = self.rnn_layers
            rnn_units = 1 if self.rnn_name == "identity" else self.rnn_units
            if self.rnn_name == "lstm":
                if self.separate:
                    return (
                        torch.zeros((num_layers, self.num_seqs, rnn_units)),
                        torch.zeros((num_layers, self.num_seqs, rnn_units)),
                        torch.zeros((num_layers, self.num_seqs, rnn_units)),
                        torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    )
                return (
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                )
            if self.separate:
                return (
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                )
            return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)

        def load(self, params):
            self.separate = params.get("separate", False)
            self.units = params["mlp"]["units"]
            self.activation = params["mlp"]["activation"]
            self.initializer = params["mlp"]["initializer"]
            self.is_d2rl = params["mlp"].get("d2rl", False)
            self.norm_only_first_layer = params["mlp"].get("norm_only_first_layer", False)
            self.value_activation = params.get("value_activation", "None")
            self.normalization = params.get("normalization", None)
            self.has_rnn = "rnn" in params
            self.has_space = "space" in params
            self.central_value = params.get("central_value", False)
            self.joint_obs_actions_config = params.get("joint_obs_actions", None)

            self.sapg_priv_cfg = params.get("sapg_priv")
            if self.sapg_priv_cfg is None:
                raise KeyError("actor_critic_sapg_priv requires `network.sapg_priv` in the train config")

            self.policy_obs_dim = int(self.sapg_priv_cfg["policy_obs_dim"])
            self.privileged_obs_dim = int(self.sapg_priv_cfg["privileged_obs_dim"])
            self.expl_embedding_dim = int(self.sapg_priv_cfg.get("expl_embedding_dim", 32))

            if self.policy_obs_dim < 0 or self.privileged_obs_dim < 0:
                raise ValueError("policy_obs_dim and privileged_obs_dim must be non-negative")

            if self.has_space:
                self.is_multi_discrete = "multi_discrete" in params["space"]
                self.is_discrete = "discrete" in params["space"]
                self.is_continuous = "continuous" in params["space"]
                if self.is_continuous:
                    self.space_config = params["space"]["continuous"]
                    self.fixed_sigma = self.space_config["fixed_sigma"]
                elif self.is_discrete:
                    self.space_config = params["space"]["discrete"]
                elif self.is_multi_discrete:
                    self.space_config = params["space"]["multi_discrete"]
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            if self.has_rnn:
                self.rnn_units = params["rnn"]["units"]
                self.rnn_layers = params["rnn"]["layers"]
                self.rnn_name = params["rnn"]["name"]
                self.rnn_ln = params["rnn"].get("layer_norm", False)
                self.is_rnn_before_mlp = params["rnn"].get("before_mlp", False)
                self.rnn_concat_input = params["rnn"].get("concat_input", False)

    def build(self, name, **kwargs):
        return A2CSAPGPrivBuilder.Network(self.params, **kwargs)
