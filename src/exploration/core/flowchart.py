import json
import warnings
from copy import copy
from typing import Dict, List

from .cohort import Cohort
from .metadata import Metadata


def metadata_from_flowchart(metadata: Metadata, flowchart_json: str) -> Metadata:
    flowchart_description = json.loads(flowchart_json)
    intermediate = flowchart_description[
        "intermediate_operations"
    ]  # type: Dict[str, Dict]
    updated_metadata = copy(metadata)
    for (_, description) in intermediate.items():
        new_cohort = metadata.get_from_description(description)
        updated_metadata = updated_metadata.add_cohort(new_cohort.name, new_cohort)
    return updated_metadata


def get_steps(flowchart_json: str) -> List[str]:
    flowchart_description = json.loads(flowchart_json)
    return flowchart_description["steps"]


class Flowchart:
    """A Flowchart is a linear list of steps. The user inputs a list of cohorts (that
    stem from the Metadata of SNIIRAM-Featuring or built cohorts using intermediate
    cohorts in the flowchart json.
    It acts as the following:
    1. Takes the first cohort as the basic starting point.
    2. For all the remaining steps, it takes the n cohort and intersect it with the cohort
    n-1.
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
        subsequent steps are the current steps.
        """
        new_steps = [input]
        new_steps.extend(self.ordered_cohorts)
        return Flowchart(new_steps)

    @staticmethod
    def from_json(metadata: Metadata, flowchart_json: str) -> "Flowchart":
        steps = get_steps(flowchart_json)  # type: List[str]
        metadata_flow_chart = metadata_from_flowchart(metadata, flowchart_json)
        new_metadata = metadata.union(metadata_flow_chart)  # type: Metadata
        return Flowchart([new_metadata.get(step) for step in steps])
