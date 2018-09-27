rm -rf MNIST
rm -rf CIFAR

mkdir MNIST
mkdir CIFAR

python run.py "MNIST" "0.1" "dense_2" "1" "3" "0.0"
# Not robust, but this is an unrealistic threat model
#python run.py "MNIST" "0.3" "dense_2" "1" "3" "0.0"

python run.py "CIFAR" "0.031" "classifier" "1" "51" "0.0"
# Using PGD based attacks to define K-NN is not robust
#python run.py "CIFAR" "0.031" "classifier" "3" "51" "0.0"
#python run.py "CIFAR" "0.031" "classifier" "4" "51" "0.0"

python ping.py

