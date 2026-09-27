import os

import imageio.v2 as imageio
import numpy as np

from envs.pushing_env import LiberoPushingEnv


# ============================================================
# Helper functions
# ============================================================

def decode_state(env, obs):
    """
    Recover absolute EEF / plate / goal positions.

    Observation:
        obs[:3]  = plate - eef
        obs[3:6] = goal - plate
    """

    plate_pos = env._get_plate_position()

    eef_pos = plate_pos - obs[:3]

    goal_pos = plate_pos + obs[3:6]

    return eef_pos, plate_pos, goal_pos


def position_controller(
    current_pos,
    target_pos,
    gain=6.0,
    max_action=0.7,
):
    """
    Simple proportional Cartesian controller.

    error = target - current

    Returns a 3D RL action in [-1, 1].
    """

    error = target_pos - current_pos

    action = gain * error

    action = np.clip(
        action,
        -max_action,
        max_action,
    )

    return action.astype(np.float32)


def record_frame(env, frames):
    frame = env.render()

    if frame is not None:
        frames.append(frame)


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        "outputs/videos",
        exist_ok=True,
    )

    env = LiberoPushingEnv(
        max_steps=400,
        action_scale=0.5,

        # Larger image only for debugging video
        observation_size=256,
    )

    obs, info = env.reset(seed=42)

    frames = []

    record_frame(env, frames)

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    (
        eef_pos,
        plate_pos,
        goal_pos,
    ) = decode_state(
        env,
        obs,
    )

    print("\n====================================")
    print("Initial geometry")
    print("====================================")

    print("EEF:")
    print(eef_pos)

    print("\nPlate:")
    print(plate_pos)

    print("\nGoal:")
    print(goal_pos)

    print(
        "\nInitial EEF -> plate distance:",
        np.linalg.norm(
            eef_pos - plate_pos
        ),
    )

    print(
        "Initial plate -> goal XY distance:",
        np.linalg.norm(
            goal_pos[:2]
            - plate_pos[:2]
        ),
    )

    # --------------------------------------------------------
    # Calculate pushing direction
    # --------------------------------------------------------

    push_direction_xy = (
        goal_pos[:2]
        - plate_pos[:2]
    )

    push_direction_xy /= (
        np.linalg.norm(
            push_direction_xy
        )
        + 1e-8
    )

    print("\nPush direction:")
    print(push_direction_xy)

    # ========================================================
    # STAGE 1
    #
    # Move ABOVE a point behind the plate.
    # ========================================================

    print("\n====================================")
    print("STAGE 1: Move behind plate")
    print("====================================")

    behind_distance = 0.05

    pre_push_xy = (
        plate_pos[:2]
        - behind_distance
        * push_direction_xy
    )

    # Stay well above the table during horizontal approach.
    hover_height = (
        plate_pos[2]
        + 0.18
    )

    hover_target = np.array(
        [
            pre_push_xy[0],
            pre_push_xy[1],
            hover_height,
        ],
        dtype=np.float32,
    )

    for step in range(100):

        (
            eef_pos,
            plate_pos,
            goal_pos,
        ) = decode_state(
            env,
            obs,
        )

        action = position_controller(
            eef_pos,
            hover_target,
            gain=6.0,
            max_action=0.7,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        record_frame(
            env,
            frames,
        )

        distance = np.linalg.norm(
            hover_target
            - eef_pos
        )

        if step % 10 == 0:

            print(
                f"step={step:03d} | "
                f"EEF-target={distance:.4f} | "
                f"reach={info['reach_distance']:.4f}"
            )

        if distance < 0.025:
            print(
                "Reached hover position."
            )
            break

    # ========================================================
    # STAGE 2
    #
    # Lower the gripper behind the plate.
    # ========================================================

    print("\n====================================")
    print("STAGE 2: Lower to pushing height")
    print("====================================")

    (
        eef_pos,
        plate_pos,
        goal_pos,
    ) = decode_state(
        env,
        obs,
    )

    # This is an initial empirical value.
    #
    # The EEF frame is above the actual finger contact point,
    # therefore it should normally remain above plate center.
    push_height = (
        plate_pos[2]
        + 0.01
    )

    push_start_target = np.array(
        [
            pre_push_xy[0],
            pre_push_xy[1],
            push_height,
        ],
        dtype=np.float32,
    )

    for step in range(80):

        (
            eef_pos,
            plate_pos,
            goal_pos,
        ) = decode_state(
            env,
            obs,
        )

        action = position_controller(
            eef_pos,
            push_start_target,
            gain=6.0,
            max_action=0.5,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        record_frame(
            env,
            frames,
        )

        distance = np.linalg.norm(
            push_start_target
            - eef_pos
        )

        if step % 10 == 0:

            print(
                f"step={step:03d} | "
                f"EEF-target={distance:.4f} | "
                f"reach={info['reach_distance']:.4f}"
            )

        if distance < 0.015:
            print(
                "Reached pushing height."
            )
            break

    # ========================================================
    # STAGE 3
    #
    # Push the plate toward the target.
    # ========================================================

    print("\n====================================")
    print("STAGE 3: Push")
    print("====================================")

    initial_goal_distance = (
        info["goal_distance"]
    )

    for step in range(180):

        (
            eef_pos,
            plate_pos,
            goal_pos,
        ) = decode_state(
            env,
            obs,
        )

        # Recalculate direction every step.
        #
        # This makes the controller closed-loop rather than
        # simply replaying a fixed trajectory.
        direction = (
            goal_pos[:2]
            - plate_pos[:2]
        )

        goal_distance = (
            np.linalg.norm(direction)
        )

        direction = (
            direction
            / (goal_distance + 1e-8)
        )

        # Closed-loop contact tracking: command a point slightly through the
        # plate along the desired direction.  Unlike a constant velocity
        # command, this stops the EEF from racing ahead after contact is lost.
        push_lead_distance = 0.025
        contact_target = np.array(
            [
                plate_pos[0] + push_lead_distance * direction[0],
                plate_pos[1] + push_lead_distance * direction[1],
                push_height,
            ],
            dtype=np.float32,
        )
        action = position_controller(
            eef_pos,
            contact_target,
            gain=8.0,
            max_action=0.55,
        )
        action[2] = np.clip(action[2], -0.25, 0.25)

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        record_frame(
            env,
            frames,
        )

        # Contact geometry for the same pre-action state printed below.
        eef_plate_xy_distance = np.linalg.norm(
            eef_pos[:2] - plate_pos[:2]
        )
        eef_plate_z_difference = (
            eef_pos[2] - plate_pos[2]
        )

        if step % 5 == 0:

            print(
                f"step={step:03d} | "
                f"goal={info['goal_distance']:.4f} | "
                f"goal_progress={info['goal_progress']:+.5f} | "
                f"reward={reward:+.4f} | "
                f"success={info['is_success']}"
            )

            print(
                "  EEF  =",
                np.round(eef_pos,4),
            )

            print(
                "  Plate =",
                np.round(plate_pos,4),
            )


            print(
                f"  EEF-plate XY="
                f"{eef_plate_xy_distance:.4f} | "
                f"Z diff="
                f"{eef_plate_z_difference:.4f}"
            )


        if terminated:

            print("\nSUCCESS!")
            break

        if truncated:
            print(
                "\nEpisode reached time limit."
            )
            break

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    final_goal_distance = (
        info["goal_distance"]
    )

    print("\n====================================")
    print("Result")
    print("====================================")

    print(
        "Initial goal distance:",
        initial_goal_distance,
    )

    print(
        "Final goal distance:",
        final_goal_distance,
    )

    print(
        "Plate displacement toward goal:",
        initial_goal_distance
        - final_goal_distance,
    )

    print(
        "Success:",
        info["is_success"],
    )

    # --------------------------------------------------------
    # Save video
    # --------------------------------------------------------

    video_path = (
        "outputs/videos/"
        "heuristic_push.mp4"
    )

    if len(frames) > 0:

        imageio.mimsave(
            video_path,
            frames,
            fps=20,
        )

        print(
            "\nVideo saved to:"
        )

        print(video_path)

    env.close()


if __name__ == "__main__":
    main()
