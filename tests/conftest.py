import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ipm-ml-matplotlib-tests')
import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(scope='session')
def experiment_output(tmp_path_factory):
    from experiments.common import Config, EXPERIMENTS
    output = tmp_path_factory.mktemp('all_experiments')
    config = Config(output, 'smoke', 1)
    with threadpool_limits(limits=1):
        for experiment in EXPERIMENTS:
            experiment(config)
    return output
