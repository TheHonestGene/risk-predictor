import logging

log = logging.getLogger(__name__)


def predict(genotype_file,trait,**kwargs):
    log_extra = kwargs.get('log_extra',{'progress':0})
    partial_progress_inc = (100-log_extra['progress'])/22
    log.info('Starting prediction for %s' % trait,extra=log_extra)
    for i in range(23):
        log_extra['progress']+=partial_progress_inc
        log.info('Computing weights for Chr%s'% i,extra=log_extra)
        import time
        time.sleep(1)
    log.info('Finished prediction',extra=log_extra)
    import random
    return random.random()