import os

from dataset import DataLoaderVal, DataLoaderTest, DataLoaderTrain
def get_training_data(rgb_dir, img_options, debug, ps=256):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None, debug, ps=ps)

def get_validation_data(rgb_dir,  debug=False, ps=256):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None, debug, ps=ps)

def get_test_data(rgb_dir,  debug=False, ps=256):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None, debug, ps=ps)
