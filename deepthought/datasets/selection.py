__author__ = 'sstober'


import collections

class DatasetMetaDB(object):
    def __init__(self, metadata, attributes):

        def multi_dimensions(n, dtype):
            """ Creates an n-dimension dictionary where the n-th dimension is of type 'type'
            """
            if n == 0:
                return dtype()
            return collections.defaultdict(lambda:multi_dimensions(n-1, dtype))

        metadb = multi_dimensions(len(attributes), list)

        for i, meta in enumerate(metadata):
            def add_entry(subdb, remaining_attributes):
                if len(remaining_attributes) == 0:
                    subdb.append(i)
                else:
                    key = meta[remaining_attributes[0]]
    #                 print remaining_attributes[0], key
                    add_entry(subdb[key], remaining_attributes[1:])
            add_entry(metadb, attributes)

        self.metadb = metadb
        self.attributes = attributes

    def select(self, selectors_dict):

        def _apply_selectors(selectors, node):
            if isinstance(node, dict):
                selected = []
                keepkeys = selectors[0]
                for key, value in node.items():
                    if keepkeys == 'all' or key in keepkeys:
                        selected.extend(_apply_selectors(selectors[1:], value))
                return selected
            else:
                return node # [node]

        selectors = []
        for attribute in self.attributes:
            if attribute in selectors_dict:
                selectors.append(selectors_dict[attribute])
            else:
                selectors.append('all')

        return sorted(_apply_selectors(selectors, self.metadb))