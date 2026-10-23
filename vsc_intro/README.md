# Running the BioID example on a KU Leuven VSC GPU

This folder contains a small PyTorch example that trains a model to find eye
positions or facial landmarks in the BioID face dataset. The `.slurm` files are
batch job scripts: Slurm schedules and runs them on a VSC cluster, rather than
running the Python work on your laptop or directly on a login node.

For KU Leuven-specific Slurm details, see the
[VSC Slurm specifics](https://docs.vscentrum.be/leuven/slurm_specifics.html).
The commands and partition names below reflect the example scripts in this
folder; check that your account and the requested partition are still available
to you before submitting jobs.

## What is in this folder?

- [`env_setup.slurm`](./env_setup.slurm) installs Miniconda under `$VSC_DATA`
  and initializes Conda for Bash. Run it once for your VSC account.
- [`install_libs.slurm`](./install_libs.slurm) creates the `rd_lab` Conda
  environment and installs Python packages, including CUDA-enabled PyTorch.
  Run it after the setup job has finished.
- [`requirements.txt`](./requirements.txt) lists the same Python packages. It
  is an alternative package list if you install dependencies manually; do not
  install both lists separately.
- [`training.py`](./training.py) loads the data, trains the
  selected model, and saves plots.
- [`base.slurm`](./base.slurm) is the example GPU training job. It needs two
  edits before use: its working-directory line is a placeholder, and it calls
  `training.py`, which is not the Python filename in this folder.
- [`BioID_FaceDatabase.zip`](./BioID_FaceDatabase.zip) contains the dataset.

## Before you start

1. Connect to a KU Leuven VSC **wICE login node** using your usual VSC access
   method. Submit Slurm jobs from the login node, not from your local computer. More info on how to connect on [this page](https://docs.vscentrum.be/compute/terminal/index.html#terminal-interface)
2. Make sure you are part of the `lp_edu_rdlab` group. You can use the command `sam-balance` to check which credit groups you are part of and the available balance (the balance is for the whole group, not just you).
3. Put the files from this folder on VSC. Use `$VSC_DATA` for the Conda
   installation, environment, dataset, and project files: it is intended for
   larger data, unlike the more limited `$VSC_HOME`.

The examples below assume you have copied the files to
`$VSC_DATA/vsc-intro`. Change that directory name in the commands and Slurm
script if you use a different location.

## Run the jobs in this order

### 1. Set up Miniconda (once)

On the wICE login node:

```bash
mkdir -p "$VSC_DATA/vsc-intro"
cd "$VSC_DATA/vsc-intro"
```
Then upload the provided files to that directory. You can use `scp` or `rsync` from your local machine if you're familiar with those commands, or use the VSC *OnDemand* web interface to upload files. More info [here](https://docs.vscentrum.be/data/transfer.html).

Once the files are in place, submit the setup job:
```bash
sbatch env_setup.slurm
```

`sbatch` prints a job ID. Check whether the job is queued or running with:

```bash
squeue -M wice -u "$USER"
```

When the setup job is no longer listed, inspect its `slurm-<jobid>.out` file in
the directory from which you submitted it. Resolve any reported error before
continuing. This script downloads Miniconda into `$VSC_DATA/miniconda3` and
adds it to your `~/.bashrc`. For a shell that was already open, load that
configuration with `source ~/.bashrc`, or open a new login session.

### 2. Create the Python environment (once, after setup)

Submit the library installation job from the same directory:

```bash
cd "$VSC_DATA/vsc-intro"
sbatch install_libs.slurm
```

Wait for the job to finish and check its `slurm-<jobid>.out` file. It creates
the `rd_lab` environment with Python 3.12 and installs NumPy, Matplotlib,
TQDM, IPython kernel support, PyTorch, and TorchVision. At the end, it prints
whether PyTorch can see CUDA. The environment and packages persist, so this step
does not need to be repeated for every training run.

### 3. Put the data in place 

From the project directory on VSC, unpack the archive:

```bash
cd "$VSC_DATA/vsc-intro"
unzip -q BioID_FaceDatabase.zip
```

The Python script's default data path is `./BioID_FaceDatabase`. The archive
creates that directory, containing matching `.pgm`, `.eye`, and `.pts` files.
Keep that layout, or pass a different directory with `--data_dir`.

### 4. Submit a GPU training job

From the directory containing `base.slurm`, submit it:

```bash
cd "$VSC_DATA/vsc-intro"
sbatch base.slurm
```

The script requests one node, one task, four CPU cores, one P100 GPU on wICE, and
15 minutes. It loads the specified wICE module, activates `rd_lab`, and runs
the Python program. The program selects CUDA when PyTorch reports it is
available; otherwise, it falls back to the CPU. The `--visualize` option saves
a labelled image for sample `0000` before training, and training then proceeds.

Monitor the queue and inspect the job output:

```bash
squeue -M wice -u "$USER"
```

Slurm writes standard output and errors to `slurm-<jobid>.out` in the
submission directory by default. After completion, the Python program saves
`BioID_0000_landmarks.png`, `training_validation_loss.png`, and
`predictions_vs_ground_truth.png` in its current working directory. Retrieve
them from `$VSC_DATA/vsc-intro`.

If a job has not started yet, it may be waiting for resources. If it finishes
with an error, read the Slurm output file first; common causes include a wrong
account, an incorrect data or project path, or the Conda environment not being
available.

## Changing the training run

The Python command accepts these options:

- `--train eyes` predicts the two eye positions (the default).
- `--train face` predicts the 20 facial landmarks.
- `--epochs 20` changes the number of training epochs (default: 10).
- `--batch_size 64` changes the batch size (default: 32).
- `--learning_rate 0.0005` changes the Adam optimizer learning rate
  (default: 0.001).
- `--val 0.4` uses 40% of the samples for validation (default: 0.2).
- `--data_dir /path/to/BioID_FaceDatabase` selects another dataset directory.
- `--visualize` saves a plot of a sample image and its annotations before
  training. `--sample_id 0001` selects another sample for that plot.

For example, update the Python command in `base.slurm` to:

```bash
python training.py --train eyes --epochs 20 --batch_size 64 \
  --learning_rate 0.0005 --visualize --val 0.4
```

Submit the edited script with `sbatch base.slurm` as before.

## Notes

- The `--clusters=wice`, `--account`, and `--partition=gpu_p100` directives in
  these scripts tell Slurm where and under which account to run the jobs. KU
  Leuven requires an explicit cluster when submitting jobs; the scripts
  already specify wICE.
- Setup and package installation are also submitted as Slurm jobs in these
  examples. They request the GPU partition even though their work is not GPU
  computation because we want pytorch to be installed with CUDA support. 
- Training plots are saved, but the Python example does not save a model
  checkpoint.
- Do not run long or GPU-intensive training directly on a login node.
