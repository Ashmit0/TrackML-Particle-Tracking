# TrackML-Particle-Tracking

## Setting Up the Virtual Environment: 
Follow the steps bellow to set up the virtual environment for the project. 

```sh
conda create --name TrackMLEnv python=3.10.8 
conda activate TrackMLEnv
```
to deactivate the evironment simply type 'deactivate' in the terminal 
```sh
conda deactivate
```

Once inside the virtual environment the next line in the terminal will install all the necessary libraries required for the project: 
```sh
pip install -r requirements.txt 
```
Once this is done install the `TrackML` package locally, rum 
```sh
pip install -e . 
```

## Install Data Locally on your device: 

Once the libraries are installed [authenticate](https://www.kaggle.com/docs/api) the kaggle api use the terminal comands to install the data locally on your device

to create a data folder 
```sh
mkdir data
```
to install specific files: 
```sh
kaggle competitions download -c trackml-particle-identification -f detectors.zip -p data/ 
```
or to install all available files: 
```sh
kaggle competitions download -c trackml-particle-identification -p data/
```