from ipie.analysis.extraction import extract_observable
import sys

estimators_filename = sys.argv[1]

qmc_data = extract_observable(estimators_filename, "S2")
print(type(qmc_data))

qmc_data = extract_observable(estimators_filename, "energy")
print(qmc_data["ETotal"])
