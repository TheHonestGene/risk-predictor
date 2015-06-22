"""
    riskpredictor
    ~~~~~~~~~~~~~
    The main module for running Risk Predictions
    :copyright: year by my name, see AUTHORS for more details
    :license: license_name, see LICENSE for more details
"""

import os
import sys
import argparse
import logging, logging.config

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s'
        },
    },
    'handlers': {
        'stdout':{
            'class' : 'logging.StreamHandler',
            'stream'  : 'ext://sys.stdout',
            'formatter': 'default',
        },
        'stderr':{
            'class' : 'logging.StreamHandler',
            'stream'  : 'ext://sys.stderr',
            'level':'ERROR',
            'formatter': 'default',
        },
    },
    'root': {
        'handlers': ['stdout','stderr'],
        'level': 'INFO',
    },
}

logging.config.dictConfig(LOGGING)
log = logging.getLogger()

def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='subcommands',description='Choose a command to run',help='Following commands are supported')                                                                  
    analysis_parser = subparsers.add_parser('run',help='Run a Risk orediction')   
    analysis_parser.set_defaults(func=run)                           
    return parser


def main(): 
    # Process arguments
    parser = get_parser()
    args = vars(parser.parse_args())
    try:
        args['func'](args)
        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception, e:
        log.exception(e)
        return 2


def run(args):
    print 'RUN RISK PREDICTION'

