import pandas as pd
from ppandas import PDataFrame


df1 = pd.read_csv("testing/numerical-1.csv")
df2 = pd.read_csv("testing/numerical-2.csv")

pd1 = PDataFrame(["Gender","Age"],df1)
pd2 = PDataFrame(["Gender","Age"],df2)

# print(pd1.bayes_net.get_cpds(node = "Gun Control"))

# print(pd1.independent_vars)
pd_join = pd1.pjoin(pd2,mismatches={"Age":'categorical'})

# print(pd1.bayes_net.get_cpds(node = "Gun Control"))
# pd_join.visualise(show_tables = True)
queryResult = pd_join.query(['Gun Control'],{"Gender":'female',"Age":'[40,60)'})
print('conditional query')
print(queryResult)

# queryResult = pd_join.query(['Gun Control'])
# print('overall query')
# print(queryResult)
# pd1.visualise(show_tables = True)