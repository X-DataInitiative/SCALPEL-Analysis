# License: BSD 3 clause

import json
import warnings
from copy import copy
from typing import List

from scalpel.core.decorators import deprecated
from .cohort import Cohort
from .cohort_collection import CohortCollection


def cohort_collection_from_cohort_flow(
    cc: CohortCollection, cohort_flow_json: str
) -> CohortCollection:
    cohort_flow_description = json.loads(cohort_flow_json)
    intermediate = cohort_flow_description[
        "intermediate_operations"
    ]  # type: Dict[str, Dict] # noqa：F821
    updated_cc = copy(cc)
    for (_, description) in intermediate.items():
        new_cohort = cc.get_from_description(description)
        updated_cc = updated_cc.add_cohort(new_cohort.name, new_cohort)
    return updated_cc


def get_steps(cohort_flow_json: str) -> List[str]:
    cohort_flow = json.loads(cohort_flow_json)
    return cohort_flow["cohorts"]


class CohortFlow:
    """A CohortFlow represents a linear list of cohorts, each cohort representing a step
    of data transformation. The user inputs a list of cohorts stemming from the cc
    file produced by SCALPEL-Extraction or list of custom cohorts defined by the user.

    It acts as the following:
    1. Takes the first cohort as the basic starting point.
    2. For all the remaining cohorts, it takes the n cohort and intersect it with the
    cohort n-1.

    Example
    -------
    cohorts = [exposed_patients, cases]
    flow = Cohortflow(cohorts)
    cohorts = flow.compute_steps(input_cohort)
    for cohort in cohorts:
        # do some statistics on resulting cohorts, such as
        fig = plt.figure()
        distribution_by_gender(cohort)
        # it will show the gender distribution of exposed patients, then the gender
        # distribution of cases. If the statistics are computed on events, they will
        # use events defined in the input_cohort.
        plt.show()

    """

    def __init__(self, cohorts: List[Cohort]):
        self._ordered_cohorts = None
        self.ordered_cohorts = cohorts
        self._compute_steps()

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.ordered_cohorts)

    @property
    def ordered_cohorts(self):
        return self._ordered_cohorts

    @ordered_cohorts.setter
    def ordered_cohorts(self, value: List[Cohort]):
        self._ordered_cohorts = value

    def _compute_steps(self):
        """Create a `CohortFlow` for the `input_cohort` cohort.
        This method outputs a new CohortFlow object (iterable). Each item of the new
        CohortFlow represents the evolution of the input_cohort cohort population when
        intersected successively with each step of the existing instance of CohortFlow.
        """
        steps_length = self.__len__()
        if steps_length == 0:
            warnings.warn("You are initiating a en Empty Flowchart.")
            self.steps = []
        elif steps_length == 1:
            self.steps = self.ordered_cohorts
        elif steps_length == 2:
            self.steps = [
                self.ordered_cohorts[0],
                self.ordered_cohorts[0].intersection(self.ordered_cohorts[1]),
            ]
        else:
            new_steps = [self.ordered_cohorts[0]]
            for step in self.ordered_cohorts[1:]:
                new_steps.append(new_steps[-1].intersection(step))
            self.steps = new_steps

    def prepend_cohort(self, input: Cohort) -> "Flowchart":
        """
        Create a new Flowchart where input is pre-appended to the existing Flowchart.
        Parameters
        ----------
        input : Cohort to be pre-appended.

        Returns
        -------
        A new Flowchart object where the new first step is the input Cohort and the
        subsequent cohorts are the current cohorts.
        """
        new_steps = [input]
        new_steps.extend(self.ordered_cohorts)
        return CohortFlow(new_steps)

    @deprecated
    @staticmethod
    def from_json(cc: CohortCollection, cohort_flow_json: str) -> "CohortFlow":
        steps = get_steps(cohort_flow_json)  # type: List[str]
        metadata_flow_chart = cohort_collection_from_cohort_flow(cc, cohort_flow_json)
        new_metadata = cc.union(metadata_flow_chart)  # type: CohortCollection
        return CohortFlow([new_metadata.get(step) for step in steps])
