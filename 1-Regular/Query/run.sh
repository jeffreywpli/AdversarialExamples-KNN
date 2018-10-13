rm -rf MNIST
rm -rf CIFAR

mkdir MNIST
mkdir CIFAR

python run.py "MNIST" "1" "3" "0.0"> "MNIST/1.txt"
python run.py "CIFAR" "1" "51" "0.0"> "CIFAR/1.txt"
python run.py "CIFAR" "2" "51" "0.0"> "CIFAR/2.txt"
python run.py "CIFAR" "3" "51" "0.0"> "CIFAR/3.txt"
python run.py "CIFAR" "4" "51" "0.0"> "CIFAR/4.txt"
python run.py "CIFAR" "5" "51" "0.0"> "CIFAR/5.txt"

