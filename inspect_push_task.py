import numpy as np

from lerobot.envs.configs import LiberoEnv


def main():
    # --------------------------------------------------
    # 1. Create the official LIBERO pushing task
    # --------------------------------------------------
    cfg = LiberoEnv(
        task="libero_goal",
        task_ids=[5],
        obs_type="pixels_agent_pos",
        observation_height=128,
        observation_width=128,
    )

    envs = cfg.create_envs(
        n_envs=1,
        use_async_envs=False,
    )

    vec_env = envs["libero_goal"][5]

    # --------------------------------------------------
    # 2. Reset
    # --------------------------------------------------
    obs, info = vec_env.reset(seed=42)

    print("\n====================================")
    print("Environment reset successfully")
    print("====================================")

    print("\nObservation keys:")
    print(obs.keys())

    # --------------------------------------------------
    # 3. Robot state
    # --------------------------------------------------
    print("\n====================================")
    print("Robot state")
    print("====================================")

    print("EEF position:")
    print(obs["robot_state"]["eef"]["pos"])

    print("\nEEF quaternion:")
    print(obs["robot_state"]["eef"]["quat"])

    print("\nGripper qpos:")
    print(obs["robot_state"]["gripper"]["qpos"])

    print("\nJoint positions:")
    print(obs["robot_state"]["joints"]["pos"])

    # --------------------------------------------------
    # 4. Access the underlying LeRobot LIBERO env
    # --------------------------------------------------
    base_env = vec_env.envs[0]

    print("\n====================================")
    print("Task information")
    print("====================================")

    print("Task name:")
    print(base_env.task)

    print("\nTask description:")
    print(base_env.task_description)

    # LeRobot wrapper
    offscreen_env = base_env._env

    # Actual LIBERO / robosuite task environment
    libero_env = offscreen_env.env

    # --------------------------------------------------
    # 5. List objects
    # --------------------------------------------------
    print("\n====================================")
    print("Objects")
    print("====================================")

    print("Movable objects:")
    for name in libero_env.objects_dict.keys():
        print("  ", name)

    print("\nFixtures:")
    for name in libero_env.fixtures_dict.keys():
        print("  ", name)

    print("\nTarget / object sites:")
    for name in libero_env.object_sites_dict.keys():
        print("  ", name)

    # --------------------------------------------------
    # 6. Object states
    # --------------------------------------------------
    print("\n====================================")
    print("Important object states")
    print("====================================")

    names_to_check = [
        "plate_1",
        "main_table_stove_front_region",
    ]

    for name in names_to_check:

        print(f"\n{name}:")

        if name not in libero_env.object_states_dict:
            print("  NOT FOUND")
            continue

        state = libero_env.object_states_dict[name]

        geom_state = state.get_geom_state()

        print("  position:")
        print(" ", geom_state["pos"])

        print("  quaternion:")
        print(" ", geom_state["quat"])

    # --------------------------------------------------
    # 7. Test zero action
    # --------------------------------------------------

    action = np.zeros((1, 7), dtype=np.float32)

    # Keep gripper open
    action[:, -1] = -1.0

    obs, reward, terminated, truncated, info = vec_env.step(action)

    print("\n====================================")
    print("After one step")
    print("====================================")

    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Info:", info)

    vec_env.close()


if __name__ == "__main__":
    main()
