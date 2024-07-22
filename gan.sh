#! /bin/bash

echo "RUNNING ${0%/*} ... "

echo "1. Creating and activating virtual environment called MMG_env."
python3 -m venv MMG_env
source MMG_env/bin/activate

echo "2. Pip install dependencies from requirements.txt"
pip3 install --upgrade pip --quiet
pip3 install -r requirements.txt --quiet
#pip3 install -f https://download.pytorch.org/whl/torch_stable.html torch==1.8.1+cu111 torchvision==0.9.1+cu111 

echo "3. Now running ${0%/*} experiments.."

############################## Train conditional DCGAN ##############################
#python3 -m gan_compare.scripts.train_gan
python3 -m gan_compare.scripts.train_gan --config_path gan_compare/configs/gan/dp_dcgan_config.yaml #--device cpu

echo "4. Finished ${0%/*}  script!!!"


