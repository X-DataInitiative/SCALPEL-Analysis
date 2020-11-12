[![CircleCI](https://circleci.com/gh/X-DataInitiative/SCALPEL-Analysis/tree/master.svg?style=shield&circle-token=77551e927f0d9f66b6c4755743d2cb7f5753395c)](https://circleci.com/gh/X-DataInitiative/SCALPEL-Analysis)
[![codecov](https://codecov.io/gh/X-DataInitiative/SCALPEL-Analysis/branch/master/graph/badge.svg?token=f78o8HzmAl)](https://codecov.io/gh/X-DataInitiative/SCALPEL-Analysis)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
![Version](https://img.shields.io/github/v/release/X-DataInitiative/SCALPEL-Analysis?include_prereleases)

# SCALPEL-Analysis

SCALPEL-Analysis is a Library part of the SCALPEL3 framework resulting from a research Partnership between [École Polytechnique](https://www.polytechnique.edu/en) & 
 [Caisse Nationale d'Assurance Maladie](https://assurance-maladie.ameli.fr/qui-sommes-nous/fonctionnement/organisation/cnam-tete-reseau)
 started in 2015 by [Emmanuel Bacry](http://www.cmap.polytechnique.fr/~bacry/) and [Stéphane Gaïffas](https://stephanegaiffas.github.io/).
 Since then, many research engineers and PhD students developped and used this framework
 to do research on SNDS data, the full list of contributors is available in [CONTRIBUTORS.md](CONTRIBUTORS.md).
 This library is based on [PySpark](https://spark.apache.org/docs/latest/api/python/pyspark.html). It provides
 useful abstractions easing cohort data analysis and manipulation. While it can be used
 as a standalone, it expects inputs formatted as the data resulting from
 SCALPEL-Extraction concept extraction, that is, a metadata.json file, tracking the
 cohorts data on disk or on HDFS:

```json
{
  "operations" : [ {
    "name" : "base_population",
    "inputs" : [ "DCIR", "MCO", "IR_BEN_R", "MCO_CE" ],
    "output_type" : "patients",
    "output_path" : "/some/path/to/base_population/data",
    "population_path" : ""
  }, {
    "name" : "drug_dispenses",
    "inputs" : [ "DCIR", "MCO", "MCO_CE" ],
    "output_type" : "acts",
    "output_path" : "/some/path/to/drug_dispenses/data",
    "population_path" : "/some/path/to/drug_dispenses/patients"
  }, ... ]
}
```

where:

- `name` contains the cohort name
- `inputs` indicates the data sources used to compute this cohort
- `ouput_type` indicates if the cohort contains only `patients` or some event type (can be custom)
- `output_path` contains the path to a parquet file containing the data
- When `output_type` is not `patients`, `output_path` is used to store events. In this case,
 `population_path` points to a parquet file containing data on the population.

In our example, the input DataFrames contain data in parquet format. If we import this 
data with PySpark and output it as strings, it should look like this :

```
base_population/data
+---------+------+-------------------+-------------------+
|patientID|gender|          birthDate|          deathDate|
+---------+------+-------------------+-------------------+
|    Alice|     2|1934-07-27 00:00:00|               null|
|      Bob|     1|1951-05-01 00:00:00|               null|
|   Carole|     2|1942-01-12 00:00:00|               null|
|    Chuck|     1|1933-10-03 00:00:00|2011-06-20 00:00:00|
|    Craig|     1|1943-07-27 00:00:00|2012-12-10 00:00:00|
|      Dan|     1|1971-10-07 00:00:00|               null|
|     Erin|     2|1924-01-12 00:00:00|               null|
+---------+------+-------------------+-------------------+
```

```
drug_dispenses/data
+---------+--------+-------+-----+------+-------------------+-------------------+
|patientID|category|groupID|value|weight|              start|                end|
+---------+--------+-------+-----+------+-------------------+-------------------+
|    Alice|exposure|   null|DrugA|   1.0|2013-08-08 00:00:00|2013-10-07 00:00:00|
|    Alice|exposure|   null|DrugB|   1.0|2012-09-11 00:00:00|2012-12-30 00:00:00|
|    Alice|exposure|   null|DrugC|   1.0|2013-01-23 00:00:00|2013-03-24 00:00:00|
|   Carole|exposure|   null|DrugB|   1.0|2010-01-25 00:00:00|2010-12-13 00:00:00|
|      Dan|exposure|   null|DrugA|   1.0|2012-11-29 00:00:00|2013-01-28 00:00:00|
|     Erin|exposure|   null|DrugC|   1.0|2010-09-09 00:00:00|2011-01-17 00:00:00|
|      Eve|exposure|   null|DrugA|   1.0|2010-04-30 00:00:00|2010-08-02 00:00:00|
+---------+--------+-------+-----+------+-------------------+-------------------+
```

```
drug_dispenses/patients
+---------+
|patientID|
+---------+
|    Alice|
|   Carole|
|      Dan|
|     Erin|
|      Eve|
+---------+
```

In these tables,

* `patientID` is a string identifying patients
* `gender` is an int indicating gender (1 for male, 2 for female ; we use the same coding as SNDS's)
* `birthDate` and `deathDate` are datetime, `deathDate` can be null
* `category` a string, used to indicate event types (drug purchase, act, drug exposure, etc.). It can be custom.
* `groupID` is a string. It is a "free" field, which is often used to perform aggregations. For example, you can use it to 
indicate drug ATC classes.
* `value` is a string, used to indicate the precise nature of the event. For example, it can
contain the CIP13 code of a drug or a ICD10 code of a disease.
* `weight` is a float, it can be used to represent quantitative information tied to the event,
such as the number of purchased boxes for drug purchase events

An event is defined by the tuple `(patientID, category, groupID, value, weight, start, end)`.
`category`, `groupID`, `value` and `weight` are flexible fields, you can fill them with
the data which best suits your needs. 

Note that the set of subjects present in `population` and `drug_dispenses` do not need to be exactly the same.

### Loading data into Cohorts
One can either create cohorts manually:

```python
from pyspark.sql import SparkSession
from scalpel.core.cohort import Cohort

spark = SparkSession.builder.appName('SCALPEL-Analysis-example').getOrCreate()
events = spark.read.parquet('/some/path/to/drug_dispenses/data')
subjects = spark.read.parquet('/some/path/to/drug_dispenses/patients')
drug_dispense_cohort = Cohort('drug_dispenses',
                              'Cohort of subjects having drug dispenses events',
                              subjects,
                              events)
```

or read import all the cohorts from a metadata.json file:

```python
from scalpel.core.cohort_collection import CohortCollection
cc = CohortCollection.from_json('/path/to/metadata.json')
print(cc.cohorts_names)  # Should print ['base_population', 'drug_dispenses']
drug_dispenses_cohort = cc.get('drug_dispenses')
base_population_cohort = cc.get('base_population')
# To access cohort data:
drug_dispenses_cohort.subjects
drug_dispenses_cohort.events
```

## Cohort manipulation

Cohorts can be manipulated easily, thanks to algebraic manipulations:

```python
# Subjects in base population who have drug dispenses
study_cohort = base_population_cohort.intersection(drug_dispenses_cohort)
# Subjects in base population who have no drug dispenses
study_cohort = base_population_cohort.difference(drug_dispenses_cohort)
# All the subjects either in base population or who have drug dispenses
study_cohort = base_population_cohort.union(drug_dispenses_cohort)
```

Note that these operations are not commutative, as 
`base_population_cohort.union(drug_dispenses_cohort)` is not equivalent to 
`drug_dispenses_cohort.union(base_population_cohort)`. Indeed, for now, these
 operations are based on `cohort.subjects`. It means that `foo` will not contain events,
 are there are no events in `base_population`, while `bar` will contain the events
 derived from `drug_dispenses_cohort`.
 
We plan to extend these manipulation in a near future to allow performing operations on
subjects and events in a single line of code.

## CohortFlow
`CohortFlow` objects can be used to track the evolution of a study population during the
cohort design process. Let us assume that you have a `CohortCollection` containing 
`base_population`, `exposed`, `cases`, respectively containing the base population of
your study, the subjects exposed to some drugs and their exposure events, the subjects 
having some disease and their disease events.

`CohortFlow` allows you to check changes in your population structure when while working
on your cohort:

```python
import matplotlib.pyplot as plt
from scalpel.stats.patients import distribution_by_gender_age_bucket
from scalpel.core.cohort_flow import CohortFlow

ordered_cohorts = [exposed, cases]

flow = CohortFlow(ordered_cohorts)
# We use 'extract_patients' as the base population
steps = flow.compute_steps(base_population)

for cohort in flow.steps:
    figure = plt.figure(figsize=(8, 4.5))
    distribution_by_gender_age_bucket(cohort=cohort, figure=figure)
    plt.show()
``` 

In this example, `CohortFlow` computes iteratively the intersection between the base 
cohort (`base_population`) and the cohorts in `ordered_cohort`, resulting in three 
steps:

* `base_population` : all subjects
* `base_population.intersection(exposed)` : exposed subjects
* `base_population.intersection(exposed).intersection(cases)` : exposed subjects who 
are cases

Calling `distribution_by_gender_age_bucket` at each step allows us to track any change 
in demographics induced by restricting the subjects to the exposed cases.

Many more plotting and statistical logging available in `scalpel.stats` can be used the
same way.

## Installation
Clone this repo and add it to the `PYTHONPATH` to use it in scripts or notebooks. To add
the library temporarily to your `PYTHONPATH`, just add 

    import sys
    sys.path.append('/path/to/the/SCALPEL-Analysis')

at the beginning of your scripts.

> **Important remark** : This software is currently in alpha stage. It should be fairly stable,
> but the API might still change and the documentation is partial. We are currently doing our best
> to improve documentation coverage as quickly as possible.

### Requirements

Python 3.6.5 or above and libraries listed in 
[requirements.txt](https://github.com/X-DataInitiative/SCALPEL-Analysis/blob/master/requirements.txt).

To create a virtual environment with `conda` and install the requirements, just run

    conda create -n <env name> python=3.5.3
    pip install -r requirements.txt

## Citation

If you use a library part of _SCALPEL3_ in a scientific publication, we would appreciate citations. You can use the following bibtex entry:

    @article{bacry2020scalpel3,
      title={SCALPEL3: a scalable open-source library for healthcare claims databases},
      author={Bacry, Emmanuel and Gaiffas, St{\'e}phane and Leroy, Fanny and Morel, Maryan and Nguyen, Dinh-Phong and Sebiat, Youcef and Sun, Dian},
      journal={International Journal of Medical Informatics},
      pages={104203},
      year={2020},
      publisher={Elsevier}
    }

 
## Contributing
The development cycle is opinionated. Each time you commit, git will
launch four checks before it allows you to finish your commit: 
1. We use [black](https://github.com/ambv/black) to format the code. 
We encourage you to install it and integrate to your code editor or IDE.
2. Some extra checks are done using Flake8
3. Testing with Nosetests
4. Coverage checks if the minimum coverage is ensured.

To activate the pre-commit hook, you just have to install the 
[requirements-dev.txt](https://github.com/X-DataInitiative/SCALPEL-Analysis/blob/master/requirements-dev.txt)
dependencies and to run:
    
    source activate <env name>
    cd SCALPEL-Analysis
    pre-commit install

To launch the tests, just run

    cd SCALPEL-Analysis
    nosetests
