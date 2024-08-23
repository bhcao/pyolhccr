# dataset structure
mkdir datasets
mkdir datasets/test
mkdir datasets/test/pot
mkdir datasets/test/vectors
mkdir datasets/train
mkdir datasets/train/pot
mkdir datasets/train/vectors
mkdir datasets/validation
mkdir datasets/validation/pot
mkdir datasets/validation/vectors

function download_pot() {
    wget https://nlpr.ia.ac.cn/databases/Download/Online/CharData/$1
    unzip $1 -d ./pot
    rm $1
}

# download train dataset
cd datasets/train
download_pot Pot1.0Train.zip
download_pot Pot1.1Train.zip
download_pot Pot1.2Train.zip

function move_pot_dataset() {
    cd pot
    files=(`ls`)
    mv ${files[*]: -$1} ../../validation/pot
    cd ../
}

# download test dataset and divided into test and validation sets
cd ../test
download_pot Pot1.1Test.zip
move_pot_dataset 30
download_pot Pot1.0Test.zip
move_pot_dataset 42
download_pot Pot1.2Test.zip
move_pot_dataset 30
cd ../

# download HIT-OR3C
wget http://www.iapr-tc11.org/dataset/OR3C_DAS2010/v1.1/OR3C/online/character.rar
unrar x character.rar
rm character.rar
cd character

rm labels.txt
mv labels ../labels
rm 117_vectors # delete error dataset

files=(`ls`)
mv ${files[*]:0:97} ../train/vectors
mv ${files[*]:97:12} ../test/vectors
mv ${files[*]:109:12} ../validation/vectors

cd ../../
rm -rf datasets/character
