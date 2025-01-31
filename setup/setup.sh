#/bash
# a bash script to setup the environment for the project
# install pip3
sudo apt-get install python3-pip -y
# install venv and create the virtual environment in the project root
python3 -m venv venv
# activate the virtual environment
source venv/bin/activate
# install packages you will need, ensuring they are being installed in the virtual env
venv/bin/pip install -U -r ../requirements.txt