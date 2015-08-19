import json
import theano
import theano.tensor as T


def sharedX(value, name=None, borrow=True, keep_on_cpu=False):
    """ Transform value into a shared variable of type floatX """
    if keep_on_cpu:
        return T._shared(theano._asarray(value, dtype=theano.config.floatX),
                         name=name,
                         borrow=borrow)

    return theano.shared(theano._asarray(value, dtype=theano.config.floatX),
                         name=name,
                         borrow=borrow)


def save_dict_to_json_file(path, dictionary):
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))


def load_dict_from_json_file(path):
    with open(path, "r") as json_file:
        return json.loads(json_file.read())
