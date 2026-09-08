from datasets import load_dataset


CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']

# Process-local cache: `load_dataset` already caches the downloaded/parsed dataset to disk
# (~/.cache/huggingface), so this dict just avoids re-touching that cache within one process.
_dataset_cache = {}


def _get_dataset():
    if 'ds' not in _dataset_cache:
        _dataset_cache['ds'] = load_dataset('fancyzhx/ag_news')
    return _dataset_cache['ds']


def get_train_data():
    split = _get_dataset()['train']
    return list(split['text']), list(split['label'])


def get_test_data():
    split = _get_dataset()['test']
    return list(split['text']), list(split['label'])
