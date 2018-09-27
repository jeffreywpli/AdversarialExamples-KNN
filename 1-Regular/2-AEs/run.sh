rm -rf MNIST
rm -rf CIFAR

mkdir MNIST
mkdir CIFAR

python run.py "MNIST" "1"
python run.py "CIFAR" "1"
python run.py "CIFAR" "2"
python run.py "CIFAR" "3"
python run.py "CIFAR" "4"
python run.py "CIFAR" "5"

