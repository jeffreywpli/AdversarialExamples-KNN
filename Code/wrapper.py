
import sys
sys.path.insert(0, "cleverhans")
from cleverhans.model import Model

# Minimal wrapper of our existing model so that we can attack them with Cleverhans

class Wrapper(Model):

    def __init__(self, model = None):
        super(Wrapper, self).__init__()

        if model is None:
            raise ValueError('model argument must be supplied.')
        
        self.model = model
        self.num_classes = 10

    def get_logits(self, x):
        return self.model.predict(x)

