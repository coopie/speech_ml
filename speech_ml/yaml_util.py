# taken from: http://stackoverflow.com/questions/6432605/any-yaml-libraries-in-python-that-support-dumping-of-long-strings-as-block-liter
"""Uttilities for pretty printing to yaml."""
import yaml
from yaml.representer import SafeRepresenter


class folded_str(str):
    pass


class literal_str(str):
    pass



def change_style(style, representer):
    def new_representer(dumper, data):
        scalar = representer(dumper, data)
        scalar.style = style
        return scalar
    return new_representer


# represent_str does handle some corner cases, so use that
# instead of calling represent_scalar directly
represent_folded_str = change_style('>', SafeRepresenter.represent_str)
represent_literal_str = change_style('|', SafeRepresenter.represent_str)

yaml.add_representer(folded_str, represent_folded_str)
yaml.add_representer(literal_str, represent_literal_str)
