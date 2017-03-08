import os
import os.path
import sys

try:
    import urllib.request as urllib
except ImportError:
    import urllib


fan_sources = [
    {
        'name': 'Cue_Target_Pairs.A-B',
        'url':
            'http://w3.usf.edu/FreeAssociation/AppendixA/'
            'Cue_Target_Pairs.A-B',
    }, {
        'name': 'Cue_Target_Pairs.C',
        'url':
            'http://w3.usf.edu/FreeAssociation/AppendixA/'
            'Cue_Target_Pairs.C',
    }, {
        'name': 'Cue_Target_Pairs.D-F',
        'url':
            'http://w3.usf.edu/FreeAssociation/AppendixA/'
            'Cue_Target_Pairs.D-F',
    }, {
        'name': 'Cue_Target_Pairs.G-K',
        'url':
            'http://w3.usf.edu/FreeAssociation/AppendixA/'
            'Cue_Target_Pairs.G-K',
    }, {
        'name': 'Cue_Target_Pairs.L-O',
        'url':
            'http://w3.usf.edu/FreeAssociation/AppendixA/'
            'Cue_Target_Pairs.L-O',
    }, {
        'name': 'Cue_Target_Pairs.P-R',
        'url':
            'http://w3.usf.edu/FreeAssociation/AppendixA/'
            'Cue_Target_Pairs.P-R',
    }, {
        'name': 'Cue_Target_Pairs.S',
        'url':
            'http://w3.usf.edu/FreeAssociation/AppendixA/'
            'Cue_Target_Pairs.S',
    }, {
        'name': 'Cue_Target_Pairs.T-Z',
        'url':
            'http://w3.usf.edu/FreeAssociation/AppendixA/'
            'Cue_Target_Pairs.T-Z',
    }
]


def report_progress(blocks_transferred, block_size, total_size):
    sys.stdout.write("\r{0}/{1} KiB".format(
        blocks_transferred * block_size // 1024, total_size // 1024))
    sys.stdout.flush()


def fetch_external(sources, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for src in sources:
        sys.stdout.write("Fetching " + src['name'] + os.linesep)
        urllib.urlretrieve(
            src['url'], os.path.join(path, src['name']), report_progress)
        sys.stdout.write(os.linesep)


if __name__ == '__main__':
    assoc_data_path = os.path.join(os.path.dirname(__file__), os.pardir,
            'association_data', 'raw_fan')

    fetch_external(fan_sources, assoc_data_path)
