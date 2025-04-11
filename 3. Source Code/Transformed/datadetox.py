import pandas as pd
import re
import unidecode
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def verifica_partidas_edi(df,edition,coluna='Mandante'):


    # Conta quantas vezes cada time foi mandante por edição
    contagem = df.groupby([edition])[coluna].value_counts()

    # Transforma em DataFrame
    contagem_df = contagem.reset_index(name='n_partidas')

    # Agora verifica, por edição, se todos os times têm o mesmo número de partidas
    verificacao = contagem_df.groupby(edition)['n_partidas'].nunique()

    # Mostra edições com mais de um número distinto de partidas (ou seja, houve desigualdade)
    edicoes_desiguais = verificacao[verificacao > 1]

    return print(f"Edições com número desigual de partidas por time: {edicoes_desiguais}")

def unificar_data_maior(df, cols_grupo, col_data='Data'):
    # Encontra a maior data para cada grupo
    datas_max = df.groupby(cols_grupo)[col_data].transform('max')
    
    # Substitui a data original pela maior
    df[col_data] = datas_max
    
    return df

#display(df_ajustado)
def mesclar_linhas_sem_conflito(grupo):
    # Verifica se todas as colunas (exceto as de agrupamento) têm valores únicos OU nulos iguais
    dados_sem_grupo = grupo.drop(columns=['Data', 'Mandante', 'Visitante', 'edition'])
    
    # Verifica se todos os valores são iguais OU nulos em cada coluna
    for col in dados_sem_grupo.columns:
        valores_unicos = grupo[col].dropna().unique()
        if len(valores_unicos) > 1:
            return grupo  # Conflito: mantém todas as linhas do jeito que estão
    
    # Caso não tenha conflito, retorna apenas uma linha com os valores "não nulos" (ou o primeiro)
    linha_mesclada = grupo.iloc[0].copy()
    for col in dados_sem_grupo.columns:
        valor = grupo[col].dropna().iloc[0] if not grupo[col].dropna().empty else None
        linha_mesclada[col] = valor
    
    return pd.DataFrame([linha_mesclada])

def ajustar_datas_semdiff(grupo,keep=True):
    grupo = grupo.sort_values('Data').reset_index(drop=True)  # Ordena por data
    for i in range(1, len(grupo)):
        if keep:
            grupo.loc[i, 'Data'] = grupo.loc[i-1, 'Data']  # Ajusta a data para a anterior
        else:
            grupo.loc[i-1, 'Data'] = grupo.loc[i, 'Data']
    return grupo

def ajustar_datas(grupo):
    grupo = grupo.sort_values('Data').reset_index(drop=True)  # Ordena por data
    for i in range(1, len(grupo)):
        diff = (grupo.loc[i, 'Data'] - grupo.loc[i-1, 'Data']).days
        if diff == 1:
            grupo.loc[i, 'Data'] = grupo.loc[i-1, 'Data']  # Ajusta a data para a anterior
    return grupo


def ajustar_datas_com_gap(grupo, gap=1, keep_max=True):
    grupo = grupo.sort_values('Data').reset_index(drop=True)  # Ordena por data
    for i in range(1, len(grupo)):
        diff = (grupo.loc[i, 'Data'] - grupo.loc[i-1, 'Data']).days
        if diff == gap:
            if keep_max:
                grupo.loc[i, 'Data'] = grupo.loc[i-1, 'Data'] 
            else:
                grupo.loc[i-1, 'Data'] = grupo.loc[i, 'Data'] 
                 # Ajusta a data para a anterior
    return grupo


#função para verificar coluna de df
def verificar_coluna(df_list, coluna):
    resultado = pd.DataFrame(columns=['Nulo', 'NaN', 'Total'])
    for df in df_list:
        resultado.loc[len(df)] = [df[coluna].isnull().sum(), 
                           df[coluna].isna().sum(),
                             len(df[coluna])]
    return resultado.reset_index(drop=True).T


# Função para detectar o padrão da data
def detectar_padrao(data_str):
    if pd.isna(data_str):
        return 'NaN'
        
    # Remover espaços
    data_str = str(data_str).strip()
    
    # Padrões comuns
    padroes = {
        r'^\d{2}/\d{2}/\d{4}$': 'DD/MM/YYYY',
        r'^\d{1,2}/\d{1,2}/\d{4}$': 'D/M/YYYY',
        r'^\d{4}-\d{2}-\d{2}$': 'YYYY-MM-DD',
        r'^\d{2}-\d{2}-\d{4}$': 'DD-MM-YYYY',
        r'^\d{1,2}-\d{1,2}-\d{4}$': 'D-M-YYYY',
        r'^\d{2}\.\d{2}\.\d{4}$': 'DD.MM.YYYY',
        r'^\d{1,2}\.\d{1,2}\.\d{4}$': 'D.M.YYYY',
        r'^\d{4}/\d{2}/\d{2}$': 'YYYY/MM/DD'
    }
    
    for regex, nome_padrao in padroes.items():
        if re.match(regex, data_str):
            return nome_padrao
            
    return 'desconhecido'

def verificar_padrao_datas(df, coluna_data='Data'):
    """
    Verifica se todas as datas na coluna seguem o mesmo padrão.
    
    Args:
        df: DataFrame contendo a coluna de datas
        coluna_data: Nome da coluna que contém as datas
        
    Returns:
        dict: Dicionário com informações sobre os padrões encontrados
    """
    import re
    from collections import Counter
    
    # Pular se a coluna já for datetime
    if pd.api.types.is_datetime64_any_dtype(df[coluna_data]):
        return {
            'padrao_unico': True,
            'tipo': 'datetime64',
            'padroes': {'datetime64': len(df)}
        }
        
    # Detectar o padrão de cada data
    padroes = df[coluna_data].apply(detectar_padrao)
    
    # Contar os padrões
    contagem_padroes = Counter(padroes)
    
    # Verificar se há apenas um padrão (excluindo NaN)
    padroes_sem_nan = {k: v for k, v in contagem_padroes.items() if k != 'NaN'}
    padrao_unico = len(padroes_sem_nan) == 1
    
    # Recuperar exemplos de cada padrão
    exemplos = {}
    for padrao in contagem_padroes.keys():
        if padrao != 'NaN':
            exemplos[padrao] = df.loc[padroes == padrao, coluna_data].iloc[0]
    
    return {
        'padrao_unico': padrao_unico,
        'padroes': dict(contagem_padroes),
        'exemplos': exemplos,
        'padrao_principal': padroes.value_counts().index[0] if not padroes.empty else None
    }


def padronizar_datas(df_list):
    """
    Padroniza todas as colunas de data nos DataFrames para datetime64[ns]
    
    Args:
        df_list: Lista de DataFrames com coluna 'Data'
        
    Returns:
        Lista de DataFrames com as datas padronizadas

    """

    import pandas as pd
    
    for i, df in enumerate(df_list):
        # Pular se já for datetime64
        if pd.api.types.is_datetime64_any_dtype(df['Data']):
            print(f"DataFrame {i}: Já é datetime64, mantido como está.")
            continue

        # Verificar o tipo de padrão predominante
        if all(df['Data'].str.contains('/', regex=False)):
            # Padrões DD/MM/YYYY ou D/M/YYYY (com separador '/')
            print(f"DataFrame {i}: Convertendo padrão com '/'")
            # Usar dayfirst=True para formatos DD/MM/YYYY e D/M/YYYY
            df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
        
        elif all(df['Data'].str.contains('-', regex=False)):
            # Padrão YYYY-MM-DD (ISO)
            print(f"DataFrame {i}: Convertendo padrão ISO com '-'")
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            
        else:
            # Tentar formato misto para casos mais complexos
            print(f"DataFrame {i}: Tentando conversão com formato misto")
            df['Data'] = pd.to_datetime(df['Data'], format='mixed', dayfirst=True, errors='coerce')
        
        # Verificar se houve valores que não puderam ser convertidos
        num_nulos = df['Data'].isna().sum()
        if num_nulos > 0:
            print(f"  Atenção: {num_nulos} valores não puderam ser convertidos")
    
    return df_list

def corrigir_times_semanticamente(lista_times, lista_referencia, modelo=None):
    """
    Compara semanticamente os nomes de times com uma lista de referência.
    
    Parâmetros:
        lista_times (list): Lista com nomes possivelmente errados ou alternativos.
        lista_referencia (list): Lista com nomes corretos (padrão).
        modelo: Um modelo SentenceTransformer. Se None, será carregado 'all-MiniLM-L6-v2'.
    
    Retorna:
        List[Tuple[str, str, float]]: Lista com (nome_original, nome_corrigido, similaridade).
    """
    if modelo is None:
        modelo = SentenceTransformer('all-MiniLM-L6-v2')
    
    emb_ref = modelo.encode(lista_referencia)
    resultados = []

    for time in lista_times:
        emb_time = modelo.encode([time])
        sims = cosine_similarity(emb_time, emb_ref)[0]
        idx_mais_similar = np.argmax(sims)
        nome_corrigido = lista_referencia[idx_mais_similar]
        similaridade = sims[idx_mais_similar]
        resultados.append((time, nome_corrigido, round(similaridade, 1)))
    
    return resultados

def exibir_times_agrupados_por_referencia(resultados, lista_referencia, limite=0.5, limite_sup=1):
    """
    Exibe os times agrupados por nome de referência com suas similaridades.
    
    Parâmetros:
        resultados: Saída da função corrigir_times_semanticamente.
        lista_referencia: Lista com os nomes corretos.
        limite: Similaridade mínima para considerar um match (default=0.5).
    """
    from collections import defaultdict

    agrupados = defaultdict(list)

    # Agrupar por nome corrigido
    for original, corrigido, score in resultados:
        if score >= limite and score < limite_sup:
            agrupados[corrigido].append(f"{original} ({score})")

    # Exibir os nomes da lista de referência em ordem
    for nome in sorted(lista_referencia):
        similares = agrupados.get(nome, [])
        if similares:
            print(f"{nome}: {', '.join(similares)}")

# Criar dicionário a partir dos resultados da correção semântica

def gerar_dicionario_mapeamento(resultados, limite=0.0):
    """
    Gera um dicionário que mapeia nomes originais para nomes corrigidos, com base na similaridade.
    
    Parâmetros:
        resultados (list): Saída da função corrigir_times_semanticamente.
        limite (float): Similaridade mínima para considerar um match.
        
    Retorna:
        dict: Mapeamento {nome_original: nome_corrigido}
    """
    mapeamento = {
        original: corrigido 
        for original, corrigido, score in resultados 
        if score >= limite
    }
    return mapeamento