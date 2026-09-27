# Clean LIBERO pushing environment

The project uses LIBERO's registered `Libero_Tabletop_Manipulation` problem
class with a project-local BDDL task. No installed package files are modified.

```bash
conda activate libero-rl
cd ~/robotics/contact_aware_pushing
python -m scripts.smoke_test_clean_env
python -m scripts.heuristic_baseline
```

Existing PPO code can keep importing `LiberoPushingEnv` from
`envs.pushing_env`. Its default observation stays 6-D and its action stays 3-D.
After `import envs`, Gym registration is also available as
`ContactAwarePushing-LIBERO-v0`.

Future experiments are opt-in, preserving old policies:

```python
env = LiberoPushingEnv(
    friction_range=(0.2, 1.0),
    include_object_velocity=True,
    include_contact_force=True,
)
```

Friction is sampled per episode but is not placed in the observation. Velocity
and contact force can be enabled independently for ablation studies.

## PPO evaluation and friction randomization

```bash
# Ten deterministic evaluations; record the first three.
MUJOCO_GL=egl python -m scripts.evaluate_ppo

# Evaluate at one exact friction value.
MUJOCO_GL=egl python -m scripts.evaluate_ppo --friction 0.3 \
  --video outputs/videos/ppo_mu_03.mp4

# Train a fresh randomized policy; save checkpoints every 50k steps.
MUJOCO_GL=egl python -m scripts.train_ppo --timesteps 1000000 \
  --friction-min 0.3 --friction-max 0.9 --save-name ppo_random_friction

# Fine-tune the fixed-friction policy under randomized friction.
MUJOCO_GL=egl python -m scripts.train_ppo \
  --resume outputs/models/ppo_fixed_friction.zip --timesteps 500000 \
  --friction-min 0.3 --friction-max 0.9 \
  --save-name ppo_random_friction_finetuned
```
