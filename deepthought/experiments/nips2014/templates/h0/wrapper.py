from __future__ import absolute_import 
import os
import deepthought.spearmint.wrapper as spearmint_wrapper

# Write a function like this called 'main'
def main(job_id, params):
  print 'Anything printed here will end up in the output directory for job #:', str(job_id);
  print params;

  print os.environ['PYTHONPATH'].split(os.pathsep)
  # yaml template and base_config are expected to be in the same directory
  meta_job_path = os.path.dirname(__file__);
  yaml_template_file = os.path.join(meta_job_path,'_template.yaml');
  base_config_path = os.path.join(meta_job_path,'_base_config.properties');
  return spearmint_wrapper.run_job(job_id, meta_job_path, yaml_template_file, base_config_path, params);
