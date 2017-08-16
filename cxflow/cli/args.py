from argparse import ArgumentParser


def get_cxflow_arg_parser() -> ArgumentParser:
    # create parser
    main_parser = ArgumentParser('cxflow')
    subparsers = main_parser.add_subparsers(help='cxflow modes')

    # create train subparser
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(subcommand='train')
    train_parser.add_argument('config_file', help='path to the config file')

    # create resume subparser
    resume_parser = subparsers.add_parser('resume')
    resume_parser.set_defaults(subcommand='resume')
    resume_parser.add_argument('config_path', help='path to the config file or the directory in which it is stored')
    resume_parser.add_argument('restore_from', nargs='?', default=None,
                               help='information passed to the net constructor (backend-specific); usually a directory '
                                    'in which the trained model is stored')

    # create predict subparser
    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(subcommand='predict')
    predict_parser.add_argument('config_path', help='path to the config file or the directory in which it is stored')
    predict_parser.add_argument('restore_from', nargs='?', default=None,
                                help='information passed to the net constructor (backend-specific); usually a '
                                     'directory in which the trained model is stored')

    # create split subparser
    split_parser = subparsers.add_parser('split')
    split_parser.set_defaults(subcommand='split')
    split_parser.add_argument('config_file', help='path to the config file')
    split_parser.add_argument('-n', '--num-splits', type=int, default=1, help='number of splits')
    split_parser.add_argument('-r', '--ratio', type=int, nargs=3, required=True, help='train, valid and test ratios')

    # create grid-search subparser
    gridsearch_parser = subparsers.add_parser('gridsearch')
    gridsearch_parser.set_defaults(subcommand='gridsearch')
    gridsearch_parser.add_argument('script', help='Script to be grid-searched')
    gridsearch_parser.add_argument('params', nargs='*', help='Params to be tested. Format: name:type=[value1,value2]. '
                                                             'Type is optional')
    gridsearch_parser.add_argument('--dry-run', action='store_true', help='Only print command output instead '
                                                                          'of executing it right away')

    # add common arguments
    for parser in [main_parser, train_parser, resume_parser, predict_parser, split_parser]:
        parser.add_argument('-v', '--verbose', action='store_true', help='increase verbosity do level DEBUG')
        parser.add_argument('-o', '--output-root', default='log', help='output directory')

    return main_parser