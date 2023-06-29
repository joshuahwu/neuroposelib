import timeit

print("Read")
print(timeit.timeit('from dappy import read'))

print("\nVis")
print(timeit.timeit('from dappy import visualization as vis'))

print("\nWrite")
print(timeit.timeit('from dappy import write'))

print("\nFeatures")
print(timeit.timeit('from dappy import features'))

print("\nAnalysis")
print(timeit.timeit('from dappy import analysis'))

print("\nRun")
print(timeit.timeit('from dappy import run'))

print("\nEmbed")
print(timeit.timeit('from dappy import embed'))

print("\nDatastruct")
print(timeit.timeit('from dappy import DataStruct as ds'))
