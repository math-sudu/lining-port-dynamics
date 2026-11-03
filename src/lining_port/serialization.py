"""Write numerical result records as JSON."""
import json
from pathlib import Path

import numpy as np


def dump(path, value):
    def convert(item):
        return item.tolist() if isinstance(item, np.ndarray) else item.item()
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n', encoding='utf-8')
