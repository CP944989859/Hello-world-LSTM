__author__ = 'CP'
import logging

def show_param(FLAGS, logger, handler):
    logger.addHandler(handler)
    logger.info('Params  below: ')
    for k in sorted(FLAGS.__dict__.keys()):
        logger.info("%s ==> %s", k, FLAGS.__dict__[k])
