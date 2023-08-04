import numpy as np

# refer to https://gist.github.com/bwhite/3726239

def rank_labels_by_predicted_scores(labels, scores):
    '''
    :param scores: num_bg × num_fg, np.array(float)
    :param labels: num_bg × num_fg, np.array(0 or 1)
    :return:
    '''
    assert isinstance(scores, np.ndarray) and isinstance(labels, np.ndarray), \
        'scores: {}, labels: {}'.format(type(scores), type(labels))
    assert scores.shape == labels.shape, '{} != {}'.format(scores.shape, labels.shape)
    indices   = (-scores).argsort(axis=1)
    first_idx = np.arange(indices.shape[0])[:, np.newaxis]
    rs_labels = labels[first_idx, indices]
    return rs_labels


def precision_multi_k(prediction, topk=(1,5,10)):
    """Computes the precision@k for the specified values of k
    :param prediction: binary matrix: num_bg × num_fg, np.array(0 or 1)
    :param topk: tuple or list
    :return:
    """
    if not isinstance(topk, (tuple, list)):
        topk = tuple(topk)
    res = []
    for k in topk:
        res.append(precision_at_k(prediction, k))
    return res

def precision_at_k(prediction, k):
    """Computes the precision@k for the specified values of k
    :param prediction: np.array(0 or 1)
    :param k: int
    :return:
    """
    assert isinstance(prediction, np.ndarray), 'invalid type {}'.format(type(prediction))
    if len(prediction.shape) != 2:
        prediction = prediction.reshape((-1, prediction.shape[0]))
    assert k >= 1
    correct_k = prediction[:, :k] != 0
    return np.round(correct_k.mean()*100,2)
    
def precision_at_k_onlyValid(prediction, flags, k):
    """Computes the precision@k for the specified values of k
    :param prediction: np.array(0 or 1)
    :param k: int
    :return:
    """
    total_num = 0
    sum = 0
    assert isinstance(prediction, np.ndarray), 'invalid type {}'.format(type(prediction))
    if len(prediction.shape) != 2:
        prediction = prediction.reshape((-1, prediction.shape[0]))
    assert k >= 1
    # correct_k = prediction[:, :k] != 0
    for i in range(prediction.shape[0]):
        sum += np.sum(prediction[i][:min(flags[i], k)])
        total_num += min(flags[i], k)
    return np.round(sum / total_num * 100,2)


def recall_at_k(prediction, k):
    """Computes the recall@k for the specified values of k
    :param prediction: np.array(0 or 1)
    :param k: int
    :return:
    """
    assert isinstance(prediction, np.ndarray), 'invalid type {}'.format(type(prediction))
    if len(prediction.shape) != 2:
        prediction = prediction.reshape((-1, prediction.shape[0]))
    assert k >= 1
    correct_k = prediction[:, :k] != 0
    recall_k  = correct_k.sum() / prediction.sum()
    recall_k  = np.round(recall_k * 100, 2)
    return recall_k


def average_precision(prediction):
    """ Score is average precision (area under PR curve)
    :param prediction: prediction: 1-d vector, np.array(0 or 1)
    :return:
    """
    assert len(prediction.shape) == 1, prediction.shape
    r = prediction != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if len(out) > 0:
        return np.mean(out)
    else:
        return 0

def mean_average_precision(prediction, k=-1):
    """
    :param prediction: binary matrix: num_bg × num_fg, np.array(0 or 1)
    :param k:
    :return:
    """
    if k > 0:
        assert k <= prediction.shape[1], \
            '{} must be less than {}'.format(k, prediction.shape[1])
        prediction = prediction[:,:k]
    ap = []
    for i in range(prediction.shape[0]):
        ap.append(average_precision(prediction[i]))
    return np.round(np.mean(ap), 2)
    
def mean_average_precision_onlyValid(prediction, flags, k=-1):
    """
    :param prediction: binary matrix: num_bg × num_fg, np.array(0 or 1)
    :param k:
    :return:
    """
    if k > 0:
        assert k <= prediction.shape[1], \
            '{} must be less than {}'.format(k, prediction.shape[1])
        prediction = prediction[:,:k]
    ap = []
    for i in range(prediction.shape[0]):
        ap.append(average_precision(prediction[i][:flags[i]]))
    return np.round(np.mean(ap), 2)


def compare_with_torchmetrics():
    from torchmetrics import RetrievalMAP
    from torch import tensor
    indexes = tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2, 0.6, 0.1, 0.2, 0.1, 0.4])
    target = tensor([False, False, True, False, True, False, True, True,\
                     False, True, True, False])
    rmap = RetrievalMAP()
    official = rmap(preds, target, indexes=indexes)
    print('official ', official)

    scores = preds.reshape(3,-1).numpy()
    labels = target.reshape(3,-1).numpy()
    rank_labels = rank_labels_by_predicted_scores(labels, scores)
    ours = mean_average_precision(rank_labels)
    print('ours ', ours)