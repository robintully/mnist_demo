import h5py
from keras.models import load_model


# Generally want this to load from s3, or pull in externally rather then expect file directly
model = load_model('app/trained_mnist.h5')
dummy_questions = h5py.File('app/dummy_questions.h5', 'r')['dataset_1'][:]


def mnist_by_index(num):
    image = dummy_questions[num].reshape(1, 28, 28, 1)
    return model.predict_classes(image)[0], image
