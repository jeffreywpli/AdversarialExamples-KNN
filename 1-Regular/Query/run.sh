rm -rf MNIST
rm -rf CIFAR

mkdir MNIST
mkdir CIFAR

python run.py "MNIST" "dense_2" "1" "3" "0.0"> "MNIST/1.txt"
python run.py "CIFAR" "classifier" "1" "51" "0.0"> "CIFAR/1.txt"
python run.py "CIFAR" "classifier" "2" "51" "0.0"> "CIFAR/2.txt"
python run.py "CIFAR" "classifier" "3" "51" "0.0"> "CIFAR/3.txt"
python run.py "CIFAR" "classifier" "4" "51" "0.0"> "CIFAR/4.txt"
python run.py "CIFAR" "classifier" "5" "51" "0.0"> "CIFAR/5.txt"

