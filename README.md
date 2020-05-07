# ppandas

This is a python package that supports joining and reasoning over multiple heterogeneous datasets without unique identifiers for each record. ppandas also handles datasets that exhibit various mismatches such as sampling bias, individual vs. aggregate-level record mismatch, numeric interval mismatch, and spatial representation mismatch.

This package accepts input data as pandas DataFrames and is backed by pgmpy.

Created by Yi (Amy) Sui and Alex Kwan under the supervision of Professor Scott Sanner at the University of Toronto.

## Related article
The package implements the Bayesian Motifs in the article ...

Use cases in this article are implemented in [experiments](https://github.com/D3Mlab/ppandas/tree/master/experiments).

## Prerequisites

This package requires the user to install pgmpy 0.1.9, networkx 2.4, matplotlib, python-interval, geopandas, geovoronoi.

```
pip install pgmpy==0.1.9
pip install networkx==2.4
pip install matplotlib
pip install python-interval
pip install geopandas
pip install geovoronoi
```

## Installing ppandas
```
pip install -i https://test.pypi.org/simple/ PPandas
```

## Tutorials
Tutorial notebooks of the package can be found in [examples](https://github.com/D3Mlab/ppandas/tree/master/examples).

## Sample code
```
pd1 = PDataFrame(["A"],df1) # Load pDataFrame with DataFrame, "A" is the independent variable
pd2 = PDataFrame(["B"],df2) # Load pDataFrame with DataFrame, "B" is the independent variable
pd_join = pd1.pjoin(pd2)   # Join pd1 and pd2 probabilistically with pd1 as the reference pDataFrame
queryResult = pd_join.query(["B"],evidence_vars={"A":0}) # query B wiht evidence A=0
```
Above is a sample code for a simple probablistic join between two datasets. The datasets are loaded in pDataFrames by specifying the independent variables and the pandas dataframe. In the join, the reference dataset is the object calling pjoin() and the secondary datset is the argument.

A special constructor is provided for aggregate-level data:
```
pd1 = PDataFrame.from_populational_data(["Age"],df1,600) # special constructor for aggregate-level data
```
The first two arguments of the special constructor are still the independent variables and DataFrame. The additional argument is the number of records.

If the two datsets contains attribute-domain mismatch, an additional argument is required to perform the probablistic join. This additional argument, 'mismatches' accepts a dictionary where the key is the attribute exhibiting mismatch that is found in both PDataFrames and the key is the type of mismatch ('numerical' or 'spatial').

```
pd1 = PDataFrame(["Gender","Age"],df1)
pd2 = PDataFrame(["Gender","Age"],df2)
pd_join = pd1.pjoin(pd2,mismatches={"Age":'numerical'})
```


## Authors

* **Yi(Amy) Sui** 

* **Alex Kwan** 

* **Alex Olson** 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
