# Train the model
cd 1-Models
./run.sh
cd ..
# Compute the AEs
cd 2-AEs
./run.sh
cd ..
# Compute the representations
cd 3-Representation
./run.sh
cd ..
# Evaluate K-NN on the training AEs
cd 4-KNN
./run.sh
cd ..
# Run attacks on the defended model for some simple configurations
cd 5-Attack
./run.sh
cd ..
# Answer Basic Questions about model and defense
cd Query
./run.sh
cd ..
