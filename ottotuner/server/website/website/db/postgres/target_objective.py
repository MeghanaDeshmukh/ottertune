#
# OtterTune - target_objective.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import logging

from ..base.target_objective import BaseThroughput
from website.types import DBMSType
from ..base.target_objective import (BaseTargetObjective, BaseThroughput, LESS_IS_BETTER,
                                     MORE_IS_BETTER)

LOG = logging.getLogger(__name__)


class ElapsedTime(BaseTargetObjective):

    def __init__(self):
        super().__init__(name='elapsed_time', pprint='Elapsed Time', unit='seconds',
                         short_unit='s', improvement=LESS_IS_BETTER)

    def compute(self, metrics, observation_time):
        return observation_time


class UnifiedHTAPMetric(BaseTargetObjective):

    def __init__(self):
        super().__init__(name='Unified_HTAP_metric', pprint='Unified HTAP Metric', unit='seconds',
                         short_unit='s', improvement=MORE_IS_BETTER)

    def compute(self, metrics, observation_time):
        variable = metrics['unified_HTAP_metric.QphH'] * metrics['unified_HTAP_metric.tpmC'];
        OLAPWorkers = metrics['unified_HTAP_metric.OLAPWorkers']
        if OLAPWorkers > 0:
            return variable/OLAPWorkers
        return variable

target_objective_list = tuple((DBMSType.POSTGRES, target_obj) for target_obj in [  # pylint: disable=invalid-name
    BaseThroughput(transactions_counter='pg_stat_database.xact_commit'),
    ElapsedTime(),
    UnifiedHTAPMetric()
])
