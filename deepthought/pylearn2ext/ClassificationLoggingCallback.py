'''
Created on Jun 17, 2014

@author: sstober
'''

import logging; 
log = logging.getLogger(__name__); 

from pylearn2.train_extensions import TrainExtension;

import numpy as np;

import theano;

from sklearn.metrics import confusion_matrix,precision_recall_fscore_support;

from pylearn2.space import CompositeSpace
from pylearn2.space import NullSpace

from deepthought.pylearn2ext.util import aggregate_classification, process_dataset;


class ClassificationLoggingCallback(TrainExtension):
    def __init__(self, dataset, model, header=None, 
                 class_prf1_channels=True, confusion_channels=True, 
#                  seq_prf1_channel=True, seq_confusion_channel=True
                 ):
        self.dataset = dataset;
        self.header = header;
        
        self.class_prf1_channels = class_prf1_channels;
        self.confusion_channels = confusion_channels;
                
        minibatch = model.get_input_space().make_theano_batch();
        self.output_fn = theano.function(inputs=[minibatch], 
                                        outputs=model.fprop(minibatch));
        
        self.data_specs = (CompositeSpace((
                                model.get_input_space(), 
                                model.get_output_space())), 
                           ("features", "targets"));
        
        if self.header is not None:
            self.channel_prefix = self.header;
        else:
            if hasattr(self.dataset, 'name'): #s  elf.dataset.name is not None:
                self.channel_prefix = self.dataset.name;
            else:
                self.channel_prefix = '';
                           
    def setup(self, model, dataset, algorithm):
        
#         print 'setup for dataset: {}\t {} '.format(dataset.name, dataset);
#         print 'self.dataset: {}\t {} '.format(self.dataset.name, self.dataset);
        
        if hasattr(self.dataset, 'get_class_labels'): 
            class_labels = self.dataset.get_class_labels();
        else:
            class_labels = ['0', '1'];
        
        # helper function
        def add_channel(name, val):
            model.monitor.add_channel(
                            name=self.channel_prefix+name,
                            ipt=None,                       # no input
                            data_specs = (NullSpace(), ''), # -> no input specs
                            val=val,
                            dataset=self.dataset,
                            );

        if self.class_prf1_channels:                                
            for class_label in class_labels:
                add_channel('_precision_'+str(class_label), 0.);
                add_channel('_recall_'+str(class_label), 0.);
                add_channel('_f1_'+str(class_label), 0.);
        
        add_channel('_f1_mean', 0.);
        
        # add channels for confusion matrix
        if self.confusion_channels:
            for c1 in class_labels:
                for c2 in class_labels:
                    add_channel('_confusion_'+c1+'_as_'+c2, 0.);

        add_channel('_seq_misclass_rate', 0.);
        add_channel('_wseq_misclass_rate', 0.);
        add_channel('_pseq_misclass_rate', 0.);
        
        add_channel('_trial_misclass_rate', 0.);
        add_channel('_wtrial_misclass_rate', 0.);
        add_channel('_ptrial_misclass_rate', 0.);
        
        add_channel('_trial_mean_f1', 0.);
        add_channel('_wtrial_mean_f1', 0.);
        add_channel('_ptrial_mean_f1', 0.);
        
        add_channel('_seq_mean_f1', 0.);
        add_channel('_wseq_mean_f1', 0.);
        add_channel('_pseq_mean_f1', 0.);
        
        
    def on_monitor(self, model, dataset, algorithm):
        
#         print 'self.dataset: {}\t {} '.format(self.dataset.name, self.dataset);
        
#         print self.dataset.X[0,0:5];
                
        y_real, y_pred, output = process_dataset(model, 
                                                 self.dataset, 
                                                 data_specs=self.data_specs, 
                                                 output_fn=self.output_fn)
        
        if self.header is not None:
            print self.header;                            

        # Compute confusion matrix
#         print classification_report(y_real, y_pred);
        conf_matrix = confusion_matrix(y_real, y_pred);
        
#         if self.dataset.name == 'test':
#             print conf_matrix;
                
        # log values in monitoring channels
        channels = model.monitor.channels;
        
        
        if hasattr(self.dataset, 'get_class_labels'): 
            class_labels = self.dataset.get_class_labels();
        else:
            class_labels = ['0', '1']; # FIXME: more flexible fallback required
        
#         p, r, f1, s = precision_recall_fscore_support(y_real, y_pred, average=None);
        p, r, f1 = precision_recall_fscore_support(y_real, y_pred, average=None)[0:3];
        
        mean_f1 = np.mean(f1);
        misclass = (y_real != y_pred).mean();
        report = [['frames', mean_f1, misclass]];
        
        channels[self.channel_prefix+'_f1_mean'].val_record[-1] = mean_f1;
        
        if self.class_prf1_channels:
            for i, class_label in enumerate(class_labels):
                channels[self.channel_prefix+'_precision_'+str(class_label)].val_record[-1] = p[i];
                channels[self.channel_prefix+'_recall_'+str(class_label)].val_record[-1] = r[i];
                channels[self.channel_prefix+'_f1_'+str(class_label)].val_record[-1] = f1[i];
        
        if self.confusion_channels:
            # add channels for confusion matrix
            for i, c1 in enumerate(class_labels):
                for j, c2 in enumerate(class_labels):
                    channels[self.channel_prefix+'_confusion_'+c1+'_as_'+c2].val_record[-1] = conf_matrix[i][j];
                    
        if self.dataset.name == 'test':
            print confusion_matrix(y_real, y_pred);

        if hasattr(self.dataset, 'sequence_partitions'):
#             print 'sequence-aggregated performance';
            
            s_real, s_pred, s_predf, s_predp = aggregate_classification(
                                                                          self.dataset.sequence_partitions,
                                                                          y_real, y_pred, output);            
            # NOTE: uses weighted version for printout
            # both, weighted and un-weighted are logged in the monitor for plotting
            
#             p, r, f1, s = precision_recall_fscore_support(s_real, s_pred, average=None);
            p, r, f1 = precision_recall_fscore_support(s_real, s_pred, average=None)[0:3];
            s_mean_f1 = np.mean(f1);
            
#             p, r, f1, s = precision_recall_fscore_support(s_real, s_predf, average=None);
            p, r, f1 = precision_recall_fscore_support(s_real, s_predf, average=None)[0:3];
            ws_mean_f1 = np.mean(f1);
            
#             p, r, f1, s = precision_recall_fscore_support(s_real, s_predp, average=None);
            p, r, f1 = precision_recall_fscore_support(s_real, s_predp, average=None)[0:3];
            ps_mean_f1 = np.mean(f1);
            
#             print classification_report(s_real, s_predf);
#             print confusion_matrix(s_real, s_predf);
            
            s_misclass = (s_real != s_pred).mean();
            ws_misclass = (s_real != s_predf).mean();
            ps_misclass = (s_real != s_predp).mean();
            
            report.append(['sequences', s_mean_f1, s_misclass]);
            report.append(['w. sequences', ws_mean_f1, ws_misclass]);
            report.append(['p. sequences', ps_mean_f1, ps_misclass]);
            
#             print 'seq misclass {:.4f}'.format(s_misclass);
#             print 'weighted seq misclass {:.4f}'.format(ws_misclass);
                        
            channels[self.channel_prefix+'_seq_misclass_rate'].val_record[-1] = s_misclass;
            channels[self.channel_prefix+'_wseq_misclass_rate'].val_record[-1] = ws_misclass;
            channels[self.channel_prefix+'_pseq_misclass_rate'].val_record[-1] = ps_misclass;
            
            channels[self.channel_prefix+'_seq_mean_f1'].val_record[-1] = s_mean_f1;
            channels[self.channel_prefix+'_wseq_mean_f1'].val_record[-1] = ws_mean_f1;
            channels[self.channel_prefix+'_pseq_mean_f1'].val_record[-1] = ps_mean_f1;
        
        if hasattr(self.dataset, 'trial_partitions'):
#             print 'trial-aggregated performance';
                        
            t_real, t_pred, t_predf, t_predp = aggregate_classification(
                                                                          self.dataset.trial_partitions,
                                                                          y_real, y_pred, output);            
            # NOTE: uses un-weighted version
            # both, weighted and un-weighted are logged in the monitor for plotting
            
#             p, r, f1, s = precision_recall_fscore_support(t_real, t_pred, average=None);
            p, r, f1 = precision_recall_fscore_support(t_real, t_pred, average=None)[0:3];
            t_mean_f1 = np.mean(f1);
            
#             p, r, f1, s = precision_recall_fscore_support(t_real, t_predf, average=None);
            p, r, f1 = precision_recall_fscore_support(t_real, t_predf, average=None)[0:3];
            wt_mean_f1 = np.mean(f1);
            
#             p, r, f1, s = precision_recall_fscore_support(t_real, t_predp, average=None);
            p, r, f1 = precision_recall_fscore_support(t_real, t_predp, average=None)[0:3];
            pt_mean_f1 = np.mean(f1);
            
#             print classification_report(t_real, t_pred);

#             if self.dataset.name == 'test':
#                 print confusion_matrix(t_real, t_predp);
            
            t_misclass = (t_real != t_pred).mean();
            wt_misclass = (t_real != t_predf).mean();
            pt_misclass = (t_real != t_predp).mean();
            
            report.append(['trials', t_mean_f1, t_misclass]);
            report.append(['w. trials', wt_mean_f1, wt_misclass]);
            report.append(['p. trials', pt_mean_f1, pt_misclass]);

#             print 'trial misclass {:.4f}'.format(t_misclass);
#             print 'weighted trial misclass {:.4f}'.format(wt_misclass);
            
            channels[self.channel_prefix+'_trial_misclass_rate'].val_record[-1] = t_misclass;
            channels[self.channel_prefix+'_wtrial_misclass_rate'].val_record[-1] = wt_misclass;
            channels[self.channel_prefix+'_ptrial_misclass_rate'].val_record[-1] = pt_misclass;
            
            channels[self.channel_prefix+'_trial_mean_f1'].val_record[-1] = t_mean_f1;
            channels[self.channel_prefix+'_wtrial_mean_f1'].val_record[-1] = wt_mean_f1;
            channels[self.channel_prefix+'_ptrial_mean_f1'].val_record[-1] = pt_mean_f1;
        
        for label, f1, misclass in report:
            print '{:>15}:  f1 = {:.3f}  mc = {:.3f}'.format(label, f1, misclass);
