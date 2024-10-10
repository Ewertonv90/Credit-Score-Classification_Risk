# Credit-Score-Classification

- Projeto de classificação de clientes de acordo com seus dados pessoais e financeiros. Dataset disponível em https://www.kaggle.com/laotse/credit-risk-dataset.
- GitHub : https://github.com/Ewertonv90/Credit-Score-Classification



# EN

Problem Statement
You are working as a data scientist in a global finance company. Over the years, the company has collected basic bank details and gathered a lot of credit-related information. The management wants to build an intelligent system to segregate the people into credit score brackets to reduce the manual efforts.

Task
Given a person’s credit-related information, build a machine learning model that can classify the credit score.

# PT-BR

Declaração do problema
Você está trabalhando como cientista de dados em uma empresa financeira global. Ao longo dos anos, a empresa coletou dados bancários básicos e reuniu muitas informações relacionadas a crédito. A gerência quer construir um sistema inteligente para segregar as pessoas em faixas de pontuação de crédito para reduzir os esforços manuais.

Tarefa
Dadas as informações relacionadas ao crédito de uma pessoa, construa um modelo de aprendizado de máquina que possa classificar a pontuação de crédito.

Bussiness Success Criteria : 85% or more

SELECT 
    cad.F0001_cod_fornecedor,
    cad.F0001_nom_fornecedor,
    tipo_serv.F0002_nom_tipo_servico,
    class_serv.F0007_cod_classificacao_tipo_servico,
    class_serv.F0007_sgl_classificacao_tipo_servico,
    class_serv.F0087_nom_classificacao_tipo_servico,
    status_forn.F0009_nom_status_fornecedor,
    sist_integrado.IN001_nom_sistema_integrado
FROM 
    soft_sup.dbo.F0001_FORNECEDOR_INT cad
LEFT JOIN 
    soft_sup.dbo.F0001_FORNECEDOR forn 
    ON cad.F0001_cod_fornecedor = forn.F0001_cod_fornecedor
LEFT JOIN 
    soft_sup.dbo.IN001_SISTEMA_INTEGRADO sist_integrado 
    ON cad.IN001_cod_sistema_integrado = sist_integrado.IN001_cod_sistema_integrado
LEFT JOIN 
    soft_sup.dbo.F0009_STATUS_FORNECEDOR status_forn 
    ON cad.F0009_cod_status_fornecedor = status_forn.F0009_cod_status_fornecedor
LEFT JOIN 
    soft_sup.dbo.F0003_FORNECEDOR_TIPO_SERVICO tipo_forn 
    ON cad.F0001_cod_fornecedor = tipo_forn.F0001_cod_fornecedor
LEFT JOIN 
    soft_sup.dbo.F0002_TIPO_SERVICO tipo_serv 
    ON tipo_forn.FO002_cod_tipo_servico = tipo_serv.FO002_cod_tipo_servico
LEFT JOIN 
    soft_sup.dbo.F0007_CLASSIFICACAO_TIPO_SERVICO class_serv 
    ON cad.F0007_cod_classificacao_tipo_servico = class_serv.F0007_cod_classificacao_tipo_servico;
