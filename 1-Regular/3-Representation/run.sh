
# Uses the model's logits as the representation
# The way this is coded is inefficient, but makes it easy to try different layers as the chosen representation

rm -rf MNIST
rm -rf CIFAR

mkdir MNIST
mkdir CIFAR

python run.py "MNIST" "dense_2"  "1"
python run.py "CIFAR" "classifier" "1"
python run.py "CIFAR" "classifier" "2"
python run.py "CIFAR" "classifier" "3"
python run.py "CIFAR" "classifier" "4"
python run.py "CIFAR" "classifier" "5"
