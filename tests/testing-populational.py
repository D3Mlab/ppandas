import pandas as pd
from ppandas import PDataFrame


df1 = pd.read_csv("testing/populational1.csv")
df1 = df1.drop(columns=["Gender"])
pd1 = PDataFrame.from_populational_data(["Age"],df1,600)
pd1.visualise(show_tables=True)