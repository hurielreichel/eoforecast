
```
sudo apt install libsodium-dev
```

```
srun -p gpu2080 --mem=40G --cpus-per-task=4 --gres=gpu:1 --time=1:00:00 --pty bash
ml purge
ml load palma/2022a
ml load foss/20224
module load R/4.2.1

#conda create --prefix ~/envs/__luz__ r-base
#CONDA_BASE=$(conda info --base)
#source $CONDA_BASE/etc/profile.d/conda.sh
#conda activate /home/h/hreichel/envs/__luz__

Rscript -e 'devtools::install_github("git@github.com:hurielreichel/eoforecast.git",
ref = "master",
auth_token = "your_token",
build=FALSE, upgrade="never",dep = TRUE,
lib = "hr_packages")'


```
