
rm -rf MNIST
rm -rf CIFAR

mkdir MNIST
mkdir CIFAR

python run.py "MNIST" "1" "0.0"

# Casual exploration of the parameters
# Pick a mode
python run.py "CIFAR" "1" "0.0"
python run.py "CIFAR" "2" "0.0"
python run.py "CIFAR" "3" "0.0"
python run.py "CIFAR" "4" "0.0"
python run.py "CIFAR" "5" "0.0"
# Explore biase
python run.py "CIFAR" "1" "0.2"
python run.py "CIFAR" "1" "0.4"
python run.py "CIFAR" "1" "0.6"
python run.py "CIFAR" "1" "0.8"
python run.py "CIFAR" "1" "0.9"
