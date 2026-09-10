import argparse

import torch


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def init_player_rnn_for_batch(player, batch_size):
    if not player.is_rnn:
        return

    default_states = player.model.get_default_rnn_state()
    resized_states = []
    for state in default_states:
        state = state.to(player.device)
        if state.size(1) == batch_size:
            resized_states.append(state)
        else:
            resized_states.append(
                torch.zeros(
                    (state.size(0), batch_size, state.size(2)),
                    dtype=state.dtype,
                    device=player.device,
                )
            )
    player.states = resized_states
