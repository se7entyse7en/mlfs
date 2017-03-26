import numpy as np


def split(dataset, training_set_percentage, validation_set_percentage=0,
          seed=0):
    np.random.seed(seed=seed)

    n = dataset['data'].shape[0]
    training_set_size = int(n / 100 * training_set_percentage)
    validation_set_size = int(n / 100 * validation_set_percentage)

    mask = list(range(n))
    np.random.shuffle(mask)
    data, target = dataset['data'][mask], dataset['target'][mask]

    return {
        'training_set': {
            'data': data[:training_set_size],
            'target': target[:training_set_size]
        },
        'validation_set': {
            'data': data[
                training_set_size:training_set_size + validation_set_size],
            'target': target[
                training_set_size:training_set_size + validation_set_size]
        },
        'test_set': {
            'data': data[training_set_size + validation_set_size:],
            'target': target[training_set_size + validation_set_size:]
        }
    }
