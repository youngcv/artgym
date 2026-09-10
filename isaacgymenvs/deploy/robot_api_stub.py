from __future__ import annotations

from isaacgymenvs.deploy.real_robot_policy_api import HandAPI, RobotObservations


class StubHand(HandAPI):
    def connect(self) -> None:
        raise NotImplementedError("Implement robot connection logic here.")

    def disconnect(self) -> None:
        raise NotImplementedError("Implement robot shutdown logic here.")

    def reset(self) -> None:
        raise NotImplementedError("Implement robot/task reset logic here.")

    def get_observations(self) -> RobotObservations:
        raise NotImplementedError(
            "Return RobotObservations(policy_obs=..., student_obs=..., expl_features=optional, episode_done=optional)."
        )

    def get_hand_joint_positions(self):
        raise NotImplementedError("Return current robot joint positions in the same DOF order as the policy.")

    def command_joint_targets(self, joint_targets) -> None:
        raise NotImplementedError("Send joint position targets to the robot.")


# Backward-compatible name for older robot-class strings.
StubRobot = StubHand
