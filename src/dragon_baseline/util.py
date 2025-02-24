from dataclasses import dataclass, asdict
import json

from pathlib import Path

@dataclass
class ArgumentsClass:
    """
    Helper class to convert dataclass to dict and json
    for dumping to file to keep track of experiments
    """
    @property
    def get_as_dict(self):
        def convert_path(obj):
            if isinstance(obj, Path):
                return str(obj)
            return obj
        # Convert dataclass to dict with path conversion
        dct = {k: convert_path(v) for k, v in asdict(self).items() 
               if self.__dataclass_fields__[k].init}
        return dct

    @property
    def json(self):
        return json.dumps(self.get_as_dict, indent=2)