import timeit

print("Read")
print(timeit.timeit("from neuroposelib import read"))

print("\nVis")
print(timeit.timeit("from neuroposelib import visualization as vis"))

print("\nWrite")
print(timeit.timeit("from neuroposelib import write"))

print("\nFeatures")
print(timeit.timeit("from neuroposelib import features"))

print("\nAnalysis")
print(timeit.timeit("from neuroposelib import analysis"))

print("\nRun")
print(timeit.timeit("from neuroposelib import run"))

print("\nEmbed")
print(timeit.timeit("from neuroposelib import embed"))

print("\nDatastruct")
print(timeit.timeit("from neuroposelib import DataStruct as ds"))
