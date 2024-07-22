#! /bin/bash

echo "RUNNING ${0%/*} ... "

echo "1. Creating and activating virtual environment called MMG_env."
python3 -m venv MMG_env
source MMG_env/bin/activate

echo "2. Pip install dependencies from requirements.txt"
#pip3 install --upgrade pip --quiet
#pip3 install -r requirements.txt --quiet
#pip3 install -f https://download.pytorch.org/whl/torch_stable.html torch==1.8.1+cu111 torchvision==0.9.1+cu111 

echo "3. Now running ${0%/*} experiments.."

### Running Experiments from https://arxiv.org/abs/2407.12669

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr0.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr0.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr0.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr1.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr1.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr1.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr2.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr2.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr2.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr3a.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr3b.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr3c.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr4.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr4.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr4.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr5a.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr5b.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr5c.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr6.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr6.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr6.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr7.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr7.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr7.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr8.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr8.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr8.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr9.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr9.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr9.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr10.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr10.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr10.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr11a.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr11b.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr11c.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr12.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr12.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr12.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr13.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr13.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr13.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr14.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr14.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr14.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr1.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr1.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr1.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr15.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr15.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr15.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr155a.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr155b.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr155c.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr70.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr70.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr70.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr90.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr90.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr90.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr100.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr100.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr100.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr156a.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr156b.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr156c.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr157a.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr157b.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr157c.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr158a.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr158b.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr158c.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr159a.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr159b.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr159c.yaml --seed 44 --dataset_path dataset16062024/

python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr160a.yaml --seed 42 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr160b.yaml --seed 43 --dataset_path dataset16062024/
python3 -m gan_compare.scripts.train_test_classifier --config_path gan_compare/configs/swin/dbr160c.yaml --seed 44 --dataset_path dataset16062024/



echo "5. Finished ${0%/*} script!!!"

