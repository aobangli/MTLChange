import json
import os

from tqdm import tqdm

from Source.DataProcess.DataProcessConfig import *

filenames = [filename for filename in os.listdir(diff_root) if filename.endswith('.json')]
for filename in tqdm(filenames):
    filepath = os.path.join(diff_root, filename)
    try:
        diff_json = json.load(open(filepath, 'r'))
    except:
        print(filename)
