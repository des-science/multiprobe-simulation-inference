import os


def get_cluster_paths():
    env = os.environ

    if "CSCS" in env.get("SITE", "") or "clariden" in str(env):
        scratch_path = "/iopsstor/scratch/cscs/athomsen"
        repo_path = "/users/athomsen/dlss/repos"
    elif "NERSC_HOST" in env:
        scratch_path = "/pscratch/sd/a/athomsen"
        repo_path = "/global/homes/a/athomsen"
    else:
        scratch_path = "/iopsstor/scratch/cscs/athomsen"
        repo_path = "/users/athomsen/dlss/repos"

    backup_path = "/capstor/store/cscs/swissai/a0158/athomsen/deep_lss"

    return scratch_path, repo_path, backup_path
