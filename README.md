# AdversarialExamples-KNN

Some notes on the required code for this library:
- nn_breaking_detection is my fork of Nicholas Carlini's Github Repo
- cleverhans in 'attacks_tf.py' there is a function called '_project_perturbation' which bounds images into [0,1].  For some experiments, we changed this to [-0.5,0.5] because that was the range the network we were using needed.  
