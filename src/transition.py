def homologacao(dataset_original, dataset_homolog, paralelismo=False, repartition_original=None, repartition_homolog=None, acuracia_aprovacao=99, coalesce_df_original=None, coalesce_df_homolog=None, arquivo=False):
    tempo_inicio = time.time()
    # # Pegar 10% do DataFrame original
    # df_10_percent = df.sample(fraction=0.1, seed=42)
    # OBRIGATORIOS: dataset_original, dataset_homolog
    # repartition_original, repartition_homolog acuracia_aprovacao=int(0.1 a 1.0), graficos=(Boolean=True ou False), arquivo=True ou False, tool=string(pandas ou spark)  
    import os
    import json
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ValueError("Para usar graficos você precisa instalar o MatPlotLib : pip install matplotlib ou usar grafico=False para não usar graficos")
    try:
        import pandas as pd
    except Exception as e:
        raise ValueError("Para usar o pandas você precisa instalar : pip install pandas ou usar tool=spark para processar com spark")
    try:
        import pyspark
    except Exception as e:
        raise ValueError("Para usar o Spark você precisa realizar a instalação ou usar tool=pandas para processar com pandas")
    from pyspark.sql.functions import col, count, when

    dataset_original = dataset_original.cache()
    dataset_homolog = dataset_homolog.cache()

    arquivo = arquivo or False
    coalesce_df_homolog = coalesce_df_homolog or None
    coalesce_df_original = coalesce_df_original or None
    paralelismo = paralelismo or None
    repartition_original = repartition_original or None
    repartition_homolog =  repartition_homolog or None
    acuracia_aprovacao = acuracia_aprovacao or 100

    if coalesce_df_homolog is not None and repartition_homolog is not None:
        raise ValueError("Não é permitido setar o o parâmetro coalesce e repartition para um mesmo dataframe de homologação")
    
    if coalesce_df_original is not None and repartition_original is not None:
        raise ValueError("Não é permitido setar o o parâmetro coalesce e repartition para um mesmo dataframe original")
    
    # repartitions
    if repartition_original is None or not repartition_original:
        pass
    else:
        dataset_original.repartition(repartition_original)

    if repartition_homolog is None or not repartition_homolog:
        pass
    else:
        dataset_homolog.repartition(repartition_homolog)


    # coalesce
    if coalesce_df_original is None or not coalesce_df_original:
        pass
    else:
        dataset_original.coalesce(coalesce_df_original)

    if coalesce_df_homolog is None or not coalesce_df_homolog:
        pass
    else:
        dataset_homolog.coalesce(coalesce_df_homolog)


    # paralelismo
    if paralelismo is None or not paralelismo:
        pass
    else:
        spark.conf.set("spark.default.parallelism", f"{paralelismo}")
       
    
    resultados = {}
    # Verificação inicial: Comparar colunas dos DataFrames
    colunas_dataset_original = set(dataset_original.columns)
    colunas_dataset_homolog = set(dataset_homolog.columns)
    
    if colunas_dataset_original != colunas_dataset_homolog:
        
        colunas_unicas_dataset_original = colunas_dataset_original - colunas_dataset_homolog
        colunas_unicas_dataset_homolog = colunas_dataset_homolog - colunas_dataset_original 
        raise ValueError(f"Os Datasets não possuem o mesmo número de colunas: \n Dataset original: {list(colunas_unicas_dataset_original)} \n Dataset de homologação: {list(colunas_unicas_dataset_homolog)}")

    # Verificação 1: Quantidade de linhas diferentes
    num_linhas_dataset_original = dataset_original.count()
    num_linhas_dataset_homolog = dataset_homolog.count()
    resultados['linhas_diferentes'] = abs(num_linhas_dataset_homolog - num_linhas_dataset_original)
    # Calcular a diferença absoluta de linhas

    # Calcular o percentual de diferença
    if num_linhas_dataset_original > 0:
        percentual_diferenca = round((num_linhas_dataset_homolog - num_linhas_dataset_original) / num_linhas_dataset_original * 100, 2)
    else:
        percentual_diferenca = round(0 , 2)  # Evitar divisão por zero

    resultados["percentual_diferenca"] = percentual_diferenca

    # Verificação 8: Dados diferentes por coluna
    dados_diferentes_por_coluna = {}

    for a in dataset_original.columns:
        if a in dataset_homolog.columns:
            # Seleciona apenas as colunas necessárias para a comparação
            df1_col = dataset_original.select(a).withColumnRenamed(a, f"{a}_df1")
            df2_col = dataset_homolog.select(a).withColumnRenamed(a, f"{a}_df2")

            # Encontra as linhas diferentes entre os dois DataFrames para a coluna atual
            diff_df1 = df1_col.exceptAll(df2_col)
            diff_df2 = df2_col.exceptAll(df1_col)

            # Conta o número de linhas diferentes para a coluna atual
            count_diff_df1 = diff_df1.count()
            count_diff_df2 = diff_df2.count()

            # Armazena o resultado na estrutura de dados
            dados_diferentes_por_coluna[a] = {
                'dataset_original': count_diff_df1,
                'dataset_homolog': count_diff_df2
            }

    resultados['dados_diferentes_por_coluna'] = dados_diferentes_por_coluna


    # Verificação 2: Quantidade de dados nulos por coluna
    nulos_dataset_original = dataset_original.select([count(when(col(col_name).isNull(), col_name)).alias(col_name) for col_name in dataset_original.columns]).collect()[0].asDict()
    nulos_dataset_homolog = dataset_homolog.select([count(when(col(col_name).isNull(), col_name)).alias(col_name) for col_name in dataset_homolog.columns]).collect()[0].asDict()

    resultados['dados_nulos_dataset_original'] = nulos_dataset_original
    resultados['dados_nulos_dataset_homolog'] = nulos_dataset_homolog

    # Verificação 3: Tipos de dados das colunas
    schema_dataset_original = {col.name: col.dataType for col in dataset_original.schema.fields}
    schema_dataset_homolog = {col.name: col.dataType for col in dataset_homolog.schema.fields}
    tipos_diferentes = {col: (schema_dataset_original[col], schema_dataset_homolog[col]) for col in schema_dataset_original if col in schema_dataset_homolog and schema_dataset_original[col] != schema_dataset_homolog[col]}

    resultados['tipos_diferentes'] = tipos_diferentes

    # Verificação 4: Colunas presentes em um DataFrame e ausentes no outro
    colunas_dataset_original = set(dataset_original.columns)
    colunas_dataset_homolog = set(dataset_homolog.columns)
    colunas_unicas_dataset_original = colunas_dataset_original - colunas_dataset_homolog
    colunas_unicas_dataset_homolog = colunas_dataset_homolog - colunas_dataset_original

    resultados['colunas_unicas_dataset_original'] = list(colunas_unicas_dataset_original)
    resultados['colunas_unicas_dataset_homolog'] = list(colunas_unicas_dataset_homolog)

    # Verificação 5: Percentual de dados nulos por coluna
    percentual_nulos_dataset_original = {col: round((nulos_dataset_original[col] / num_linhas_dataset_original) * 100 , 2) for col in nulos_dataset_original}
    percentual_nulos_dataset_homolog = {col: round((nulos_dataset_homolog[col] / num_linhas_dataset_homolog) * 100, 2) for col in nulos_dataset_homolog}

    resultados['percentual_nulos_dataset_original'] = percentual_nulos_dataset_original
    resultados['percentual_nulos_dataset_homolog'] = percentual_nulos_dataset_homolog

    # Verificação 6: Saúde geral do DataFrame (média de nulos por coluna)
    media_nulos_dataset_original = sum(nulos_dataset_original.values()) / len(nulos_dataset_original)
    media_nulos_dataset_homolog = sum(nulos_dataset_homolog.values()) / len(nulos_dataset_homolog)

    resultados['saude_geral_dataset_original'] = round(100 - media_nulos_dataset_original, 2)
    resultados['saude_geral_dataset_homolog'] = round(100 -  media_nulos_dataset_homolog, 2)

    # Verificação 7: Linhas exclusivas em cada DataFrame
    exclusivas_dataset_original = dataset_original.exceptAll(dataset_homolog)
    exclusivas_dataset_homolog = dataset_homolog.exceptAll(dataset_original)

    exclusivas_dataset_original = exclusivas_dataset_original.cache()
    exclusivas_dataset_original = exclusivas_dataset_original.cache()

    amostra_exclusivas_dataset_original = exclusivas_dataset_original.limit(10).collect()
    amostra_exclusivas_dataset_homolog = exclusivas_dataset_homolog.limit(10).collect()

    resultados['amostra_exclusivas_dataset_original'] = [row.asDict() for row in amostra_exclusivas_dataset_original]
    resultados['amostra_exclusivas_dataset_homolog'] = [row.asDict() for row in amostra_exclusivas_dataset_homolog]

    path = f"{os.getcwd()}"
    path = path.replace("\\", "/")

    if len(resultados['amostra_exclusivas_dataset_homolog']) == 0 and len(resultados['amostra_exclusivas_dataset_original']) == 0:
        print("DataFrame original e de Homologação possuem todas as linhas identicas!")
    
    if arquivo == True and len(resultados) > 0:
        with open(f"{path}/resultados.json", 'w') as json_file:
            json.dump(resultados, json_file, indent=4)

    tempo_fim = time.time()
    tempo_decorrido = tempo_fim - tempo_inicio
    tempo_em_minutos = tempo_decorrido / 60.00
    resultados["tempo_processamento"] = round(tempo_decorrido, 2)

    # Mostrar resultados
    for chave, valor in resultados.items():
        if chave == "amostra_exclusivas_dataset_homolog" or "amostra_exclusivas_dataset_original":
            pass
        print(f"{chave}: {valor}")

    # Transformar amostra de linhas exclusivas dos DataFrames e mostrar se houver
    if len(resultados['amostra_exclusivas_dataset_original']) > 0:
        print("\n")
        amostra_exclusivas_dataset_original = resultados['amostra_exclusivas_dataset_original']
        df_amostra_exclusivas_dataset_original = spark.createDataFrame(amostra_exclusivas_dataset_original)
        df_amostra_exclusivas_dataset_original = df_amostra_exclusivas_dataset_original.cache()
        if arquivo == True and len(resultados['amostra_exclusivas_dataset_original']) > 0:
            with open(f"{path}/amostra_diferenca_original.json", 'w') as json_file:
                json.dump(resultados["amostra_exclusivas_dataset_original"], json_file, indent=4)
        print( f"Amostra de dados divergentes que estão no dataset original e não no de homologação:")
        df_amostra_exclusivas_dataset_original.show(10)
    else: 
        pass

    if len(resultados['amostra_exclusivas_dataset_homolog']) > 0:
        print("\n")
        amostra_exclusivas_dataset_homolog = resultados['amostra_exclusivas_dataset_homolog']
        df_amostra_exclusivas_dataset_homolog = spark.createDataFrame(amostra_exclusivas_dataset_homolog)
        df_amostra_exclusivas_dataset_homolog = df_amostra_exclusivas_dataset_homolog.cache()
        if arquivo == True and len(resultados['amostra_exclusivas_dataset_original']) > 0:
            with open(f"{path}/amostra_diferenca_homolog.json", 'w') as json_file:
                json.dump(resultados["amostra_exclusivas_dataset_homolog"], json_file, indent=4)
        print( f"Amostra de dados divergentes que estão no dataset de homologação e não no original:")
        df_amostra_exclusivas_dataset_homolog.show(10)
    else: 
        pass
    
    if resultados["percentual_diferenca"] < acuracia_aprovacao:
        raise ValueError(f"DATASET REPROVADO DE ACORDO COM ACURACIA SOLICITADA DE {acuracia_aprovacao}% \n RESULTADO OBTIDO: {percentual_diferenca}%")
     
    return resultados
