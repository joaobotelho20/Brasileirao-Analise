#Bibliotecas
import os
import pandas as pd
import json



#Funções
def unir_xlsx_em_df(pasta):
    """
    Lê todos os arquivos .xlsx de uma pasta e os une em um único DataFrame,
    mantendo apenas o cabeçalho do primeiro arquivo.

    Parâmetros:
    - pasta (str): Caminho da pasta com os arquivos .xlsx.

    Retorna:
    - df_final (pd.DataFrame): DataFrame unificado.
    """
    arquivos = sorted([f for f in os.listdir(pasta) if f.endswith(".xlsx")])
    
    if not arquivos:
        print("Nenhum arquivo .xlsx encontrado.")
        return pd.DataFrame()

    lista_dfs = []

    for i, arquivo in enumerate(arquivos):
        caminho_arquivo = os.path.join(pasta, arquivo)

        if i == 0:
            # Primeiro arquivo: lê com cabeçalho
            df = pd.read_excel(caminho_arquivo)
        else:
            # Restante: ignora o cabeçalho
            df = pd.read_excel(caminho_arquivo, header=None, skiprows=1)
            df.columns = lista_dfs[0].columns  # aplica o cabeçalho do primeiro

        lista_dfs.append(df)

    df_final = pd.concat(lista_dfs, ignore_index=True)
    return df_final




def ler_json(caminho):
    """
    Lê arquivo JSON e entrega em uma estrutura DataFrame.

    Parâmetros:
    - caminho (str): Caminho completo do arquivo .json.
      Exemplo: r'C:\\Users\\...\\arquivo.json'

    Retorna:
    - df (pd.DataFrame): DataFrame com os dados carregados do JSON.
    """
    with open(caminho, 'r', encoding='utf-8') as arquivo:
        dados = json.load(arquivo)
    df = pd.DataFrame(dados)
    return df

def expandir_json(df: pd.DataFrame, prefixo: str) -> pd.DataFrame:
    """
    Expande a coluna 'pontuacao_geral_{prefixo}' de um DataFrame, que contém dicionários ou JSONs,
    em várias colunas separadas, prefixando os nomes com o valor de 'prefixo'.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada contendo a coluna com dados em formato dict/JSON.
        prefixo (str): Prefixo que define o nome da coluna a ser expandida e será usado nas novas colunas.

    Retorna:
        pd.DataFrame: Novo DataFrame com as colunas expandidas e renomeadas com o prefixo.
    """

    coluna_json = df[f'pontuacao_geral_{prefixo}']
    colunas_expandida = pd.json_normalize(coluna_json)
    colunas_expandida.columns = [f'{prefixo}_{col}' for col in colunas_expandida.columns]

    return colunas_expandida