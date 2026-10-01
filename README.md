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

Additional observations are opt-in, preserving old policies:

```python
env = LiberoPushingEnv(
    friction_range=(0.2, 1.0),
    include_object_velocity=True,
    include_contact_force=True,
)
```

Friction is sampled per episode but is not placed in the observation.

## Formal v1 observations and history

The formal observation modes are:

| Mode | Single-step contents | Dimension |
| --- | --- | ---: |
| `pos6` | Relative positions | 6 |
| `posvel9` | Relative positions + object XYZ velocity | 9 |
| `posvel_force11` | Relative positions + velocity + contact-force XY | 11 |

`posforce9` and `posvelforce12` remain available only for reproducing legacy
models. Formal force experiments exclude the Z force because gravity and table
support can dominate that component.

`--history-length 1` is memoryless. `--history-length 4` supplies the policy
with:

```text
[o_t, a_t-1, o_t-1, a_t-2, o_t-2, a_t-3, o_t-3]
```

The resulting dimensions are 45 for `posvel9` and 53 for
`posvel_force11`. At reset, all observation slots contain the initial
observation and all past-action slots contain zeros.

Fresh PPO runs use separate 128-by-128 ReLU actor and critic networks.

## PPO training and evaluation

```bash
# Train a v1 policy with XY contact force and four-frame history.
MUJOCO_GL=egl python -m scripts.train_ppo \
  --env-version v1 \
  --observation-mode posvel_force11 \
  --history-length 4 \
  --timesteps 1000000 \
  --run-name ppo_posvel_force11_k4_seed42 \
  --save-name ppo_v1 \
  --friction-min 0.3 --friction-max 0.9 \
  --seed 42

# Reconstruct the same observation/history pipeline from config.json.
MUJOCO_GL=egl python -m scripts.evaluate_ppo \
  --run-dir outputs/runs/ppo_posvel_force11_k4_seed42 \
  --evaluation-name test_seeds_4000_4019 \
  --episodes 20 \
  --video-episodes 3 \
  --seed 4000 \
  --friction-min 0.3 --friction-max 0.9
```
