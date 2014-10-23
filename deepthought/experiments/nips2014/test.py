import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import os;
import deepthought;
DATA_PATH = os.path.join(deepthought.DATA_PATH, 'rwanda2013rhythms');
MODEL_PATH = os.path.join(deepthought.OUTPUT_PATH, 'nips2014', 'models', 'h0');
OUTPUT_PATH = os.path.join(deepthought.OUTPUT_PATH, 'nips2014', 'figures', 'h0');
print 'data path  : {}'.format(DATA_PATH);
print 'model path : {}'.format(MODEL_PATH);
print 'output path: {}'.format(OUTPUT_PATH);



# test with subject 4
# WARNING: code seems to be broken due to library update!
from deepthought.experiments.nips2014.scripts.generate_plots import load_results;
from deepthought.pylearn2ext.util import process_dataset;
path4 = os.path.join(MODEL_PATH, '4', 'best');
train, model = load_results(path4);
dataset = train.algorithm.monitoring_dataset['test'];
y_real, y_pred, output = process_dataset(model, dataset);


# subject 4 analysis
from deepthought.experiments.nips2014.scripts.generate_plots import multi_level_accuracy_analysis;
multi_level_accuracy_analysis(y_real, y_pred);


from deepthought.pylearn2ext.util import aggregate_classification
t_real, t_pred, t_predf, t_predp = aggregate_classification(dataset.trial_partitions, y_real, y_pred, output);  
multi_level_accuracy_analysis(t_real, t_pred);