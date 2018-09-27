# AdversarialExamples-KNN

Some notes on the required code for this project:
- Code/nn_breaking_detection is our fork (https://github.com/GDPlumb/nn_breaking_detection) of Nicholas Carlini's repository (https://github.com/carlini/nn_breaking_detection)
- Code/cleverhans is from https://github.com/tensorflow/cleverhans.  For some of our experiments, we modified the file 'attacks_tf.py' so that the function '_project_perturbation' projected images into [-0.5,0.5] rather than [0,1].
