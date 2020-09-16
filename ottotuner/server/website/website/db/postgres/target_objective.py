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

class tpmC(BaseTargetObjective):
    def __init__(self):
        super().__init__(name='HTAP_tpmC', pprint='HTAP tpmC', unit='seconds',
                         short_unit='s', improvement=MORE_IS_BETTER)
    def compute(self, metrics, observation_time):
        variable = metrics['unified_HTAP_metric.tpmC'];
        return variable

class QphH(BaseTargetObjective):

    def __init__(self):
        super().__init__(name='HTAP_QphH', pprint='HTAP QphH', unit='seconds',
                         short_unit='s', improvement=MORE_IS_BETTER)

    def compute(self, metrics, observation_time):
        variable = metrics['unified_HTAP_metric.QphH'];
        return variable

class RankSum_tpmC_QphH(BaseTargetObjective):
    def __init__(self):
        super().__init__(name='HTAP_RankSum_tQ', pprint='HTAP RankSum tQ', unit='seconds',
                         short_unit='s', improvement=MORE_IS_BETTER)
    def compute(self, metrics, observation_time):
        ### wi = 2(n+1 - i) / n(n+1);  n: total attributes; i: Rank;
        ### tpmC: i=0; QphH: i=1;
        w1 = 2*(3-0)/6;
        w2 = 2*(3-1)/6;
        variable = (metrics['unified_HTAP_metric.QphH']*w2) + (w1*metrics['unified_HTAP_metric.tpmC'])
        return variable

class RankSum_QphH_tpmC(BaseTargetObjective):
    def __init__(self):
        super().__init__(name='HTAP_RankSum_Qt', pprint='HTAP RankSum Qt', unit='seconds',
                         short_unit='s', improvement=MORE_IS_BETTER)
    def compute(self, metrics, observation_time):
        ### wi = 2(n+1 - i) / n(n+1);  n: total attributes; i: Rank;
        ### tpmC: i=0; QphH: i=1;
        w1 = 2*(3-0)/6;
        w2 = 2*(3-1)/6;
        variable = (metrics['unified_HTAP_metric.QphH']*w1) + (w2*metrics['unified_HTAP_metric.tpmC'])
        return variable

target_objective_list = tuple((DBMSType.POSTGRES, target_obj) for target_obj in [  # pylint: disable=invalid-name
    BaseThroughput(transactions_counter='pg_stat_database.xact_commit'),
    ElapsedTime(),
    UnifiedHTAPMetric(),
    tpmC(),
    QphH(),
    RankSum_tpmC_QphH(),
    RankSum_QphH_tpmC()
])
