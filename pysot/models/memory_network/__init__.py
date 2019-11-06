
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.memory_network.key_generator import KeyGenerator
from pysot.models.memory_network.memory_base import MemoryBase

KEY_GENERATORS = {
                'KeyGenerator': KeyGenerator,
                }

MEMORY_BASES = {
              'MemoryBase': MemoryBase,
             }

def get_key_generator(name, **kwargs):
    return KEY_GENERATORS[name](**kwargs)

def get_memory_base(name, **kwargs):
    return MEMORY_BASES[name](**kwargs)