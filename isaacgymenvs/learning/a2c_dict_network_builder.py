import torch
import torch.nn as nn

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.network_builder import NetworkBuilder


class A2CBuilder(NetworkBuilder):
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

            self.input_networks_actor = nn.ModuleDict()
            self.input_networks_critic = nn.ModuleDict()
            self.input_out_sizes = {}
            self.last_privileged_latent = None

            for input_name, input_config in self.input_params.items():
                actor_layers = []
                critic_layers = []

                has_cnn = "cnn" in input_config
                has_mlp = "mlp" in input_config

                member_input_shape = torch_ext.shape_whc_to_cwh(input_shape[input_name])

                if has_cnn:
                    cnn_config = input_config["cnn"]
                    cnn_args = {
                        "ctype": cnn_config["type"],
                        "input_shape": member_input_shape,
                        "convs": cnn_config["convs"],
                        "activation": cnn_config["activation"],
                        "norm_func_name": self.normalization,
                    }
                    actor_layers.append(self._build_conv(**cnn_args))
                    if self.separate:
                        critic_layers.append(self._build_conv(**cnn_args))

                next_input_shape = self._calc_input_size(
                    member_input_shape, actor_layers[-1] if has_cnn else None
                )

                if has_mlp:
                    mlp_config = input_config["mlp"]
                    mlp_args = {
                        "input_size": next_input_shape,
                        "units": mlp_config["units"],
                        "activation": mlp_config["activation"],
                        "norm_func_name": mlp_config.get("normalization", None),
                        "dense_func": torch.nn.Linear,
                        "d2rl": mlp_config.get("d2rl", False),
                        "norm_only_first_layer": mlp_config.get("norm_only_first_layer", False),
                    }
                    actor_layers.append(self._build_mlp(**mlp_args))
                    if self.separate:
                        critic_layers.append(self._build_mlp(**mlp_args))
                    next_input_shape = mlp_config["units"][-1]

                self.input_networks_actor[input_name] = nn.Sequential(*actor_layers)
                self.input_networks_critic[input_name] = nn.Sequential(*critic_layers)
                self.input_out_sizes[input_name] = next_input_shape

            self.actor_privileged_encoder = None
            self.critic_privileged_encoder = None
            self.privileged_latent_dim = 0
            if self.has_privileged_encoder:
                priv_encoder_input_size = sum(
                    self.input_out_sizes[name] for name in self.privileged_input_names
                )
                self.actor_privileged_encoder = self._build_privileged_encoder(priv_encoder_input_size)
                if self.separate:
                    self.critic_privileged_encoder = self._build_privileged_encoder(priv_encoder_input_size)

                priv_units = self.privileged_encoder_cfg.get("mlp", {}).get("units", [])
                self.privileged_latent_dim = priv_units[-1] if priv_units else priv_encoder_input_size

            direct_input_size = sum(
                self.input_out_sizes[name] for name in self.direct_input_names
            )
            in_mlp_shape = direct_input_size + self.privileged_latent_dim

            if len(self.units) == 0:
                out_size = in_mlp_shape
            else:
                out_size = self.units[-1]

            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                    if self.rnn_concat_input:
                        rnn_in_size += in_mlp_shape
                else:
                    rnn_in_size = in_mlp_shape
                    in_mlp_shape = self.rnn_units

                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                "input_size": in_mlp_shape,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": torch.nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config["mu_activation"])
                mu_init = self.init_factory.create(**self.space_config["mu_init"])
                self.sigma_act = self.activations_factory.create(self.space_config["sigma_activation"])
                sigma_init = self.init_factory.create(**self.space_config["sigma_init"])

                if self.fixed_sigma == "fixed":
                    self.sigma = nn.Parameter(
                        torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                        requires_grad=True,
                    )
                elif self.fixed_sigma == "coef_cond":
                    self.sigma_ids = kwargs["coef_ids"]
                    self.sigma_id_idx = kwargs["coef_id_idx"]
                    self.sigma = nn.Parameter(
                        torch.zeros(
                            len(self.sigma_ids),
                            actions_num,
                            requires_grad=True,
                            dtype=torch.float32,
                        ),
                        requires_grad=True,
                    )
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

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

            if self.has_privileged_encoder:
                encoder_init = self._get_privileged_initializer()
                self._apply_linear_init(self.actor_privileged_encoder, encoder_init)
                if self.separate:
                    self._apply_linear_init(self.critic_privileged_encoder, encoder_init)

        def _build_privileged_encoder(self, input_size):
            mlp_cfg = self.privileged_encoder_cfg.get("mlp", {})
            units = mlp_cfg.get("units", [])
            if not units:
                return nn.Identity()

            encoder_args = {
                "input_size": input_size,
                "units": units,
                "activation": mlp_cfg.get("activation", self.activation),
                "norm_func_name": mlp_cfg.get("normalization", self.normalization),
                "dense_func": torch.nn.Linear,
                "d2rl": mlp_cfg.get("d2rl", False),
                "norm_only_first_layer": mlp_cfg.get("norm_only_first_layer", False),
            }
            return self._build_mlp(**encoder_args)

        def _get_privileged_initializer(self):
            initializer_cfg = self.privileged_encoder_cfg.get("mlp", {}).get("initializer")
            if initializer_cfg is None:
                initializer_cfg = self.initializer
            return self.init_factory.create(**initializer_cfg)

        def _apply_linear_init(self, module, initializer):
            if module is None:
                return
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    initializer(layer.weight)
                    if getattr(layer, "bias", None) is not None:
                        torch.nn.init.zeros_(layer.bias)

        def _prepare_obs(self, obs):
            obs = {**obs}
            for input_name, input_config in self.input_params.items():
                if "cnn" in input_config and len(obs[input_name].shape) == 4:
                    obs[input_name] = obs[input_name].permute((0, 3, 1, 2))
            return obs

        def _encode_branch(self, obs, input_networks, privileged_encoder):
            direct_features = []
            privileged_features = []

            for input_name in self.input_names:
                processed = input_networks[input_name](obs[input_name])
                processed = processed.contiguous().view(processed.size(0), -1)
                if input_name in self.privileged_input_name_set:
                    privileged_features.append(processed)
                else:
                    direct_features.append(processed)

            fused_features = list(direct_features)
            privileged_latent = None
            if privileged_features:
                privileged_features = torch.cat(privileged_features, dim=-1)
                privileged_latent = privileged_encoder(privileged_features)
                fused_features.append(privileged_latent)

            if not fused_features:
                raise RuntimeError("actor_critic_dict needs at least one direct input or one privileged encoder input")

            if len(fused_features) == 1:
                return fused_features[0], privileged_latent
            return torch.cat(fused_features, dim=-1), privileged_latent

        def forward(self, obs_dict):
            obs = self._prepare_obs(obs_dict["obs"])
            states = obs_dict.get("rnn_states", None)
            seq_length = obs_dict.get("seq_length", 1)
            dones = obs_dict.get("dones", None)
            bptt_len = obs_dict.get("bptt_len", 0)

            if self.separate:
                a_out, a_latent = self._encode_branch(
                    obs,
                    self.input_networks_actor,
                    self.actor_privileged_encoder,
                )
                c_out, _ = self._encode_branch(
                    obs,
                    self.input_networks_critic,
                    self.critic_privileged_encoder,
                )
                self.last_privileged_latent = a_latent

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
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    c_out = c_out.reshape(num_seqs, seq_length, -1)

                    a_out = a_out.transpose(0, 1)
                    c_out = c_out.transpose(0, 1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0, 1)

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
                        raise NotImplementedError(
                            "actor_critic_dict does not yet support fixed_sigma='coef_cond' with dict observations"
                        )
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                    return mu, sigma, value, states
            else:
                out, latent = self._encode_branch(
                    obs,
                    self.input_networks_actor,
                    self.actor_privileged_encoder,
                )
                self.last_privileged_latent = latent

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
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0, 1)
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
                        raise NotImplementedError(
                            "actor_critic_dict does not yet support fixed_sigma='coef_cond' with dict observations"
                        )
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

            self.input_params = params.get("inputs", params.get("input_preprocessors"))
            if self.input_params is None:
                raise KeyError("actor_critic_dict requires `network.inputs` (or legacy `input_preprocessors`) in the train config")

            self.input_names = list(self.input_params.keys())
            self.privileged_encoder_cfg = params.get("privileged_encoder")
            self.has_privileged_encoder = self.privileged_encoder_cfg is not None
            if self.has_privileged_encoder:
                self.privileged_input_names = list(self.privileged_encoder_cfg.get("input_names", []))
                if not self.privileged_input_names:
                    raise ValueError("privileged_encoder.input_names must list at least one dict observation key")
                unknown_inputs = set(self.privileged_input_names) - set(self.input_names)
                if unknown_inputs:
                    raise KeyError(
                        f"privileged_encoder.input_names contains unknown inputs: {sorted(unknown_inputs)}"
                    )
            else:
                self.privileged_input_names = []

            self.privileged_input_name_set = set(self.privileged_input_names)
            self.direct_input_names = [
                name for name in self.input_names if name not in self.privileged_input_name_set
            ]

            self.value_activation = params.get("value_activation", "None")
            self.normalization = params.get("normalization", None)
            self.has_rnn = "rnn" in params
            self.has_space = "space" in params
            self.central_value = params.get("central_value", False)
            self.joint_obs_actions_config = params.get("joint_obs_actions", None)

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
        return A2CBuilder.Network(self.params, **kwargs)
