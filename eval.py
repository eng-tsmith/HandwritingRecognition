from keras.models import model_from_json
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils.visualize_util import plot
import numpy
import os


# load json and create model
experiment = "5_epochs"
file_path_model = os.path.join(os.getcwd(), "output/", experiment)
file_path_weigths = os.path.join(file_path_model, "weights/")

json_file = open(os.path.join(file_path_model, "model.json"), 'r')
loaded_model_json = json_file.read()
json_file.close()

print("Loading model...")
loaded_model = model_from_json(loaded_model_json)
print("Loaded model from disk")

# load weights into new model
print("Loading weights...")
loaded_model.load_weights(os.path.join(file_path_weigths, "weights04.h5"))
print("Loaded weights to model")

# Optimizer
# clipnorm seems to speeds up convergence
clipnorm = 5
lr = 0.001
sgd = SGD(lr=lr, decay=3e-7, momentum=0.9, nesterov=True, clipnorm=clipnorm)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# evaluate loaded model on test data
loaded_model.compile(optimizer=rms, loss={'ctc': lambda y_true, y_pred: y_pred})

plot(loaded_model, to_file=os.path.join(file_path_model, 'model_eval.png'))

# score = loaded_model.evaluate(X, Y, verbose=0)  #TODO
# print("Score: ", score)