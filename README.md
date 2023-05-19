# ENSAE-ELTDM-final-project
Final project done for the course "Elements Logiciels pour le Traitement des Données Massives" taught at ENSAE 2023

### Activate virtual environnment and install requirements
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Compile the Cython extension module
```
python3 setup.py build_ext --inplace
```

### Line profiling of training with SGD
```
kernprof -l -v MF_scripts/MF_profiling.py
```

### Generate benchmark charts for execution speed
```
cd MF_scripts
python benchmark.py
```
