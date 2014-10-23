'''
Created on Apr 17, 2014

@author: sstober
'''

subject_groups = [[0,  9],  # 180 + 240
                  [1, 10],
                  [2, 11],
                  [6,  3],
                  [7,  4],
                  [8,  5],
                  [   12]   # no 180 for this one                   
                ];
                
def merge_subject_groups(groups):
    return sum(groups[0:4], []); # uses overloaded + operator for list