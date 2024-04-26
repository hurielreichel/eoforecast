## NO_2 Forecasting using Sentinel 5P and Weather Forecast Data

![Spacetime animation created by the eoforecast R package](vignettes/images/spacetime-animation.gif)

This repo is the main repo to my MSc Thesis in Spatial Data Science at the University of Münster. 

The thesis code has been designed as an R package, which should handle dependecies **per se** to a certain extent. In [vignettes/](vignettes) you will find presentatations, files and the [vignettes/main-thesis.qmd](most reproducible way to run the whole thesis). 

In [R/](R directory), you will find the R package modules, where you can look into more detail into the Neural Network Architecture, ETL, and coding in general. 

The project itself is about forecasting NO_2 pollution in both space and time, making use of a network called CNN-LSTM, that inherits both space and time structures to make this possible. You will see a lot about spatial datacubes in there as well, and extreme 3D/4D matrices handling. To have a look at the final thesis version, go [vignettes/main-thesis.pdf](here).

### Package Installation

The package has only been tested in Linux, most specifically Ubuntu 22.04 LTS. There, you will probably need:

```
sudo apt install libsodium-dev
```

Some runs were also testes in the University of Münster HPC called Palma, where this was also necessary:

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
```

and then, finally, to install the package, you can basically use the following command, that should abstract the whole R package dependencies installation: 

```
Rscript -e 'devtools::install_github("git@github.com:hurielreichel/eoforecast.git",
ref = "master",
auth_token = "your_token",
build=FALSE, upgrade="never",dep = TRUE,
lib = "hr_packages")'
```
