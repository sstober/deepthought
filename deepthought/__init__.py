from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from pylearn2.utils.string_utils import preprocess

# load system-specific settings 
# and make them available at top-level namespace
DATA_PATH = preprocess('${DEEPTHOUGHT_DATA_PATH}')
OUTPUT_PATH = preprocess('${DEEPTHOUGHT_OUTPUT_PATH}')

# Remove this from the top-level namespace.
del preprocess