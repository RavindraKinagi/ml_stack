assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="assembled")
assembler_features = assembler.transform(df).select("assembled", "Outcome")

assembler_features.toPandas().head()
scaler = StandardScaler(inputCol='assembled', outputCol='features')
df_scaled = scaler.fit(assembler_features).transform(assembler_features).select("features", "Outcome")

df_scaled.toPandas().head()

