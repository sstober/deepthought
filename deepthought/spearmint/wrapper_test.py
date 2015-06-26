'''
Created on May 11, 2014

@author: sstober
'''
import os
import deepthought.spearmint.wrapper as spearmint_wrapper

if __name__ == '__main__':
    params = {
              'learning_rate' : [0.01]
              }
    
    job_id = 'job1';
#     meta_job_path = os.path.dirname(__file__);
    meta_job_path = '/Users/sstober/git/deepbeat/deepbeat/spearmint/audiomostly/h0_input_1bar_nophase/';
    yaml_template_file = os.path.join(meta_job_path,'_template.yaml');
    # yaml_template_file = '/Users/sstober/work/develop/code-repos/deepthought-dev.git/deepthought/experiments/rwanda2013scae/_template.yaml'
    base_config_path = os.path.join(meta_job_path,'_base_config.properties');
    result = spearmint_wrapper.run_job(
                                       job_id, 
                                       meta_job_path, 
                                       yaml_template_file, 
                                       base_config_path, 
                                       params,
                                       );
    
    print result;