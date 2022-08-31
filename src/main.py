#!/usr/bin/env python
"""
run.py - A CYBORG-SAL-Collab file
Copyright (C) 2021  Zach Carmichael
"""
if __name__ == '__main__':
    import argparse
    from src.config import C

    parser = argparse.ArgumentParser(
        description='Entry script to generic model search experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # global config
    config_dict = C.as_dict()
    parser.add_argument(
        '-B', '--backbone', help='The name of the pre-trained backbone to use.',
        choices=['AlexNet', 'ResNet50', 'ViT', 'VGG'],
        default=config_dict.pop('BACKBONE'),
    )
    parser.add_argument(
        '-L', '--loss', help='The name of the loss to use.',
        choices=['CE', 'Psych-RT'],
        default=config_dict.pop('LOSS'),
    )



    for param, default in config_dict.items():
        param_name = param.lower().replace('_', '-')
        type_ = type(default)
        if type_ is bool:
            param_name = (f'--{param_name}' if not default else
                          f'--no-{param_name}')
            kwargs = {'action': 'store_true', 'dest': param.lower()}
        else:
            param_name = f'--{param_name}'
            kwargs = {'type': type_, 'default': default}
        parser.add_argument(
            param_name, help=f'{param}', **kwargs
        )
    args = parser.parse_args()

    # now, update global config with user-supplied arguments
    for param in C.keys():
        value = getattr(args, param.lower())
        setattr(C, param, value)

    # delay larger imports until this point (to e.g. speed up displaying help)
    from src.run import run

    run(C)
