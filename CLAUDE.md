AGENTS.md

# Task Execution

This repository runs on a Slurm cluster. All training/inference tasks **must** be submitted to compute nodes via `sbatch`, not run directly on the login node.

## Submitting tasks

```bash
# Submit a job script
sbatch your_script.sh

# Check job status
squeue -u $USER

# Cancel a job
scancel <job_id>

# View job output
cat slurm-<job_id>.out
```

## Writing sbatch scripts

Training scripts (e.g. `examples/ppo_trainer/*.sh`) should be wrapped in an sbatch script with appropriate Slurm directives. Example:

```bash
#!/bin/bash
#SBATCH --job-name=verl-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --partition=<partition>
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Run training
bash examples/ppo_trainer/run_moonlight16b_a3b_gsm8k_megatron.sh
```

See `examples/slurm/ray_on_slurm.slurm` for a multi-node Ray-based example.

**Do NOT** run GPU-intensive training or inference directly on login/head nodes.