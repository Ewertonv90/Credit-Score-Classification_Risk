# Exemplo de DataFrames com colunas e dados nulos
data1 = [
    ("Ana", 28, "ana@example.com", "1234567890", "Rua A", "Cidade A"),
    ("Bruno2", 40, "bruno@example.com", "987654321", "Rua B", "Cidade B"),
    ("Carlos", 30, None, "123123123", "Rua F", "Cidade Z"),
    ("Daniela", 28, "daniela@example.com", None, "Rua D", "Cidade D"),
    ("Eduardo", 38, "eduardo@example.com", "456456456", None, "Cidade Z"),
    ("Fernanda", 22, "fernanda@example.com", "789789789", "Rua F", None),
    ("Gustavo", None, "gustavo@example.comm", "321321321", "Rua G", "Cidade G"),
    ("Helena", 27, None, "654654654", "Rua H", "Cidade H"),
    ("Igor", 29, "igor@example.comm", "987987987", "Rua I", "Cidade I"),
    ("Julia", None, "julia@example.comm", "321654987", "Rua J", "Cidade J"),
    ("Karina", 32, None, "123789456", None, "Cidade K"),
    ("Luis", 31, "luis@example.com", None, "Rua L", "Cidade L"),
    ("Marina", 26, "marina@example.com", "654987321", "Rua M", None),
    ("Nina1", None, "nina@example.comm", "789123456", "Rua N", "Cidade Z"),
    ("Otavio", 33, "otavio@example.com", "123456123", "Rua O", "Cidade O"),
    ("Paula", 24, "paula@example.com", "456789789", "Rua P", "Cidade P"),
    ("Rafaelo", None, "rafael@example.com", "789456123", "Rua Q", "Cidade Q"),
    ("Sofia", 23, "sofia@example.com", "123123789", "Rua N", "Cidade R"),
    ("Tiago", 34, "tiago@example.com", "456123456", "Rua S", "Cidade S"),
    ("Ursula", 28, "ursula@example.comm", "789789123", None, "Cidade Z"),
    ("Victoria", None, "victor@example.com", "321321654", "Rua U", "Cidade U"),
    ("Wagner", 29, None, "654654321", "Rua U", "Cidade V"),
    ("Yarara", 27, "yara@example.com", "987987654", "Rua W", "Cidade W")
]

data2 = [
    ("Ana", 25, "ana@example.com", "123456789", "Rua A", "Cidade A"),
    ("Bruno", None, "bruno@example.com", "987654321", "Rua B", "Cidade B"),
    ("Carlos", 30, None, "123123123", "Rua C", "Cidade C"),
    ("Daniela", 28, "daniela@example.com", None, "Rua D", "Cidade D"),
    ("Eduardo", 35, "eduardo@example.com", "456456456", None, "Cidade E"),
    ("Fernanda", 22, "fernanda@example.com", "789789789", "Rua F", None),
    ("Gustavo", None, "gustavo@example.com", "321321321", "Rua G", "Cidade G"),
    ("Helena", 27, None, "654654654", "Rua H", "Cidade H"),
    ("Igor", 29, "igor@example.com", "987987987", "Rua I", "Cidade I"),
    ("Julia", None, "julia@example.com", "321654987", "Rua J", "Cidade J"),
    ("Karina", 32, None, "123789456", None, "Cidade K"),
    ("Luis", 31, "luis@example.com", None, "Rua L", "Cidade L"),
    ("Marina", 26, "marina@example.com", "654987321", "Rua M", None),
    ("Nina", None, "nina@example.com", "789123456", "Rua N", "Cidade N"),
    ("Otavio", 33, "otavio@example.com", "123456123", "Rua O", "Cidade O"),
    ("Paula", 24, "paula@example.com", "456789789", "Rua P", "Cidade P"),
    ("Rafael", None, "rafael@example.com", "789456123", "Rua Q", "Cidade Q"),
    ("Sofia", 23, "sofia@example.com", "123123789", "Rua R", "Cidade R"),
    ("Tiago", 34, "tiago@example.com", "456123456", "Rua S", "Cidade S"),
    ("Ursula", 28, "ursula@example.com", "789789123", None, "Cidade T"),
    ("Victor", None, "victor@example.com", "321321654", "Rua U", "Cidade U"),
    ("Wagner", 29, None, "654654321", "Rua V", "Cidade V"),
    ("Yara", 27, "yara@example.com", "987987654", "Rua W", "Cidade W"),
    ("Zeca", 40, "zeca@example.com", "123123123", "Rua Z", "Cidade Z"),
    ("Ana2", 45, "ana2@example.com", "123456789", "Rua A2", "Cidade A2"),
    ("Bruno2", None, "bruno2@example.com", "987654321", "Rua B2", "Cidade B2"),
    ("Carlos2", 30, None, "123123123", "Rua C2", "Cidade C2"),
    ("Daniela2", 28, "daniela2@example.com", None, "Rua D2", "Cidade D2"),
    ("Eduardo2", 35, "eduardo2@example.com", "456456456", None, "Cidade E2"),
    ("Fernanda2", 22, "fernanda2@example.com", "789789789", "Rua F2", None),
    ("Gustavo2", None, "gustavo2@example.com", "321321321", "Rua G2", "Cidade G2"),
    ("Helena2", 27, None, "654654654", "Rua H2", "Cidade H2"),
    ("Igor2", 29, "igor2@example.com", "987987987", "Rua I2", "Cidade I2"),
    ("Julia2", None, "julia2@example.com", "321654987", "Rua J2", "Cidade J2"),
    ("Karina2", 32, None, "123789456", None, "Cidade K2"),
    ("Luis2", 31, "luis2@example.com", None, "Rua L2", "Cidade L2"),
    ("Marina2", 26, "marina2@example.com", "654987321", "Rua M2", None),
    ("Nina2", None, "nina2@example.com", "789123456", "Rua N2", "Cidade N2")
]

columns2_diferença = ["nome", "email", "telefone", "endereco", "cidade"]

data2_diferença_colunas = [
    ("Ana", "ana@example.com", "123456789", "Rua A", "Cidade A"),
    ("Bruno", "bruno@example.com", "987654321", "Rua B", "Cidade B"),
    ("Carlos", None, "123123123", "Rua C", "Cidade C"),
    ("Daniela", "daniela@example.com", None, "Rua D", "Cidade D"),
    ("Eduardo", "eduardo@example.com", "456456456", None, "Cidade E"),
    ("Fernanda", "fernanda@example.com", "789789789", "Rua F", None),
    ("Gustavo", "gustavo@example.com", "321321321", "Rua G", "Cidade G"),
    ("Helena", None, "654654654", "Rua H", "Cidade H"),
    ("Igor", "igor@example.com", "987987987", "Rua I", "Cidade I"),
    ("Julia", "julia@example.com", "321654987", "Rua J", "Cidade J"),
    ("Karina", None, "123789456", None, "Cidade K"),
    ("Luis", "luis@example.com", None, "Rua L", "Cidade L"),
    ("Marina", "marina@example.com", "654987321", "Rua M", None),
    ("Nina", "nina@example.com", "789123456", "Rua N", "Cidade N"),
    ("Otavio", "otavio@example.com", "123456123", "Rua O", "Cidade O"),
    ("Paula", "paula@example.com", "456789789", "Rua P", "Cidade P"),
    ("Rafael", "rafael@example.com", "789456123", "Rua Q", "Cidade Q"),
    ("Sofia", "sofia@example.com", "123123789", "Rua R", "Cidade R"),
    ("Tiago", "tiago@example.com", "456123456", "Rua S", "Cidade S"),
    ("Ursula", "ursula@example.com", "789789123", None, "Cidade T"),
    ("Victor", "victor@example.com", "321321654", "Rua U", "Cidade U"),
    ("Wagner", None, "654654321", "Rua V", "Cidade V"),
    ("Yara", "yara@example.com", "987987654", "Rua W", "Cidade W"),
    ("Zeca", "zeca@example.com", "123123123", "Rua X", "Cidade X"),
    ("Ana2", "ana2@example.com", "123456789", "Rua A2", "Cidade A2"),
    ("Bruno2", "bruno2@example.com", "987654321", "Rua B2", "Cidade B2"),
    ("Carlos2", None, "123123123", "Rua C2", "Cidade C2"),
    ("Daniela2", "daniela2@example.com", None, "Rua D2", "Cidade D2"),
    ("Eduardo2", "eduardo2@example.com", "456456456", None, "Cidade E2"),
    ("Fernanda2", "fernanda2@example.com", "789789789", "Rua F2", None),
    ("Gustavo2", "gustavo2@example.com", "321321321", "Rua G2", "Cidade G2"),
    ("Helena2", None, "654654654", "Rua H2", "Cidade H2"),
    ("Igor2", "igor2@example.com", "987987987", "Rua I2", "Cidade I2"),
    ("Julia2", "julia2@example.com", "321654987", "Rua J2", "Cidade J2"),
    ("Karina2", None, "123789456", None, "Cidade K2"),
    ("Luis2", "luis2@example.com", None, "Rua L2", "Cidade L2"),
    ("Marina2", "marina2@example.com", "654987321", "Rua M2", None),
    ("Nina2", "nina2@example.com", "789123456", "Rua N2", "Cidade N2")
]

columns1 = ["nome", "idade", "email", "telefone", "endereco", "cidade"]
columns2 = ["nome", "idade","email", "telefone", "endereco", "cidade"]

# data2_diferença_colunas , data_2 
dataset_original = spark.createDataFrame(data1, columns1)
dataset_homolog = spark.createDataFrame(data2, columns2)


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
