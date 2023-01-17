# Creating venv
python3.10 -m venv venv

# Activating venv
source venv/bin/activate

# Deactivate
deactivate

# Getting standard gitignore
wget https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore

# Installing jupyter in venv
pip3 install jupyter notebook

# Adding the environment to ipython
ipython kernel install --user --name=venv
