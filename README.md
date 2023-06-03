
```
sudo apt install libsodium-dev
```

```
srun -p gpu2080 --mem=40G --cpus-per-task=4 --gres=gpu:1 --time=1:00:00 --pty bash
module purge
module load palma/2021a Miniconda3/4.9.2
module load GCC/10.3.0
module load OpenMPI/4.1.1
module load R

conda create --prefix ~/envs/__luz__ r-base
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /home/h/hreichel/envs/__luz__

Rscript -e 'devtools::install_github("git@github.com:hurielreichel/eoforecast.git",
ref = "master",
auth_token = "your_token",
build=FALSE, upgrade="never",dep = TRUE)'
```
