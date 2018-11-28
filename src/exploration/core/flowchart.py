import json
from copy import copy
from typing import List

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

    def __init__(self, steps: List[Cohort]):
        self.steps = steps  # type: List[Cohort]

    @staticmethod
    def from_json(metadata: Metadata, flowchart_json: str) -> "Flowchart":
        steps = get_steps(flowchart_json)  # type: List[str]
        metadata_flow_chart = metadata_from_flowchart(metadata, flowchart_json)
        new_metadata = metadata.union(metadata_flow_chart)  # type: Metadata
        return Flowchart([new_metadata.get(step) for step in steps])

    def create_flowchart(self, input: Cohort) -> "Flowchart":
        """Create a flowchart for the input."""
        new_steps = [input.intersection(self.steps[0])]  # type: List[Cohort]
        for step in self.steps[1:]:
            new_steps.append(new_steps[-1].intersection(step))
        return Flowchart(new_steps)

    def __iter__(self):
        return iter(self.steps)
