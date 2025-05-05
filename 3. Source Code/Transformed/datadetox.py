import pandas as pd
import re
import unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from json import JSONDecodeError

def distribuir_pontos(df, mandante, visitante, pontos_mandante='Pontos Mandante', pontos_visitante='Pontos Visitante'):
    """Distribui pontos de partidas entre times mandantes e visitantes conforme resultados.
    
    Atribui pontos às equipes baseado no resultado do confronto:
    - Em caso de empate: 1 ponto para cada time
    - Em caso de vitória: 3 pontos para o vencedor, 0 para o perdedor
    
    Args:
        df (pd.DataFrame): DataFrame contendo os resultados das partidas
        mandante (str): Nome da coluna com os resultados/placares do time mandante
        visitante (str): Nome da coluna com os resultados/placares do time visitante
        pontos_mandante (str, optional): Nome da coluna para os pontos do mandante. 
            Defaults to 'Pontos Mandante'.
        pontos_visitante (str, optional): Nome da coluna para os pontos do visitante.
            Defaults to 'Pontos Visitante'.
    
    Returns:
        pd.DataFrame: DataFrame original com as duas novas colunas de pontos adicionadas/modificadas
    
    """
    df[pontos_mandante] = np.where(
        df[mandante] == df[visitante], 1,
        np.where(df[mandante] > df[visitante], 3, 0))
    
    df[pontos_visitante] = np.where(
        df[mandante] == df[visitante], 1,
        np.where(df[mandante] < df[visitante], 3, 0))
    
    return df

def ler_json_as_dict(arquivo):
    """
    Lê um arquivo JSON e retorna como dicionário Python.
    
    Args:
        arquivo (str): Caminho para o arquivo JSON
        
    Returns:
        dict: Dicionário com os dados do JSON ou None em caso de erro
    """
    dados = None  # Inicializa como None para caso ocorram erros
    
    try:
        with open(arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{arquivo}' não encontrado")
    except JSONDecodeError as jde:
        print(f"Erro: JSON mal formatado no arquivo '{arquivo}': {jde}")
    except Exception as e:
        print(f"Erro inesperado ao ler '{arquivo}': {e}")
    
    return dados


def normalizar_texto(texto):
    if pd.isna(texto):
        return texto
    # Transforma em string, caixa baixa
    texto = str(texto).lower()
    # Remove acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    # Remove espaços extras
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def arredondar_quase_15min(dt,min=1,div=15):
    minuto = dt.minute
    resto = minuto % div
    if resto == div - min:
        return dt + pd.Timedelta(minutes=min)
    elif resto == min:
        return dt - pd.Timedelta(minutes=min)
    else:
        return dt

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

def pont_acumulada(df, season):
    """
    Calcula o aproveitamento acumulado dos times mandantes e visitantes
    ao longo de uma temporada específica de um campeonato de futebol.

    O aproveitamento é definido como a razão entre os pontos acumulados
    e o número máximo de pontos possíveis até a rodada atual (rodada * 3).

    Args:
        df (pandas.DataFrame): 
            DataFrame contendo os dados dos jogos. Deve conter, no mínimo,
            as seguintes colunas:
            - 'edition': Identificador da temporada.
            - 'Rodada': Número da rodada do jogo.
            - 'Mandante': Nome do time mandante.
            - 'Visitante': Nome do time visitante.
            - 'Pontos Mandante': Pontos obtidos pelo mandante no jogo.
            - 'Pontos Visitante': Pontos obtidos pelo visitante no jogo.

        season (int or str): 
            Temporada a ser considerada para o cálculo. Deve corresponder
            aos valores da coluna 'edition'.

    Returns:
        pandas.DataFrame: 
            O DataFrame original com duas novas colunas adicionadas:
            - 'AP_Mand Acumulados': Aproveitamento acumulado do mandante.
            - 'AP_Vis Acumulados': Aproveitamento acumulado do visitante.

    Observações:
        - A função atualiza o DataFrame linha a linha, mantendo controle
          do total de pontos acumulados por time.
        - Ideal para análises de desempenho rodada a rodada.
        - Para grandes volumes de dados, considere alternativas vetorizadas.
    """

    mask = df['edition'] == season
    df_season = df[mask].copy()

    pontos_acumulados = {time: 0 for time in df_season['Mandante'].unique()}

    for idx, row in df_season.iterrows():
        mandante = row['Mandante']
        visitante = row['Visitante']
        pt_mandante = row['Pontos Mandante']
        pt_visitante = row['Pontos Visitante']
        rodada = row['Rodada']
        max_pontos = rodada * 3

        # Atualiza pontos acumulados
        pontos_acumulados[mandante] += pt_mandante
        pontos_acumulados[visitante] += pt_visitante

        # Calcula aproveitamento
        ap_mandante = pontos_acumulados[mandante] / max_pontos
        ap_visitante = pontos_acumulados[visitante] / max_pontos

        df.loc[idx, 'AP_Mand Acumulados'] = ap_mandante
        df.loc[idx, 'AP_Vis Acumulados'] = ap_visitante

    return df

def yield_last_xrounds(df: pd.DataFrame, x: int, season: int | str) -> pd.DataFrame:
    """
    Calcula o rendimento dos clubes como mandantes nos últimos 'x' jogos, para uma temporada específica.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados das partidas, com colunas como 'temporada', 'Rodada',
        'Mandante', 'Visitante', 'Pontos Mandante' e 'Pontos Visitante'.

    x : int
        Número de rodadas anteriores a serem consideradas para calcular o rendimento.

    season : int ou str
        Temporada (ano) a ser filtrada no DataFrame.

    Retorna:
    --------
    pd.DataFrame
        DataFrame original com uma nova coluna chamada 'Yield last {x} rounds',
        indicando o rendimento dos clubes mandantes com base nos últimos 'x' jogos.
    """
    df_season = df[df['temporada'] == season].copy()

    clubes = list(dict.fromkeys(list(df_season['Mandante'])))
    dicionario: dict[str, list[int]] = {clube: [] for clube in clubes}

    df_season = df_season.sort_values(by=['temporada', 'Rodada'], ascending=[True, True])

    for idx, row in df_season.iterrows():
        mandante = row['Mandante']
        visitante = row['Visitante']
        pt_mandante = row['Pontos Mandante']
        pt_visitante = row['Pontos Visitante']

        if dicionario[mandante]:
            rendimento = sum(dicionario[mandante]) / (len(dicionario[mandante]) * 3)
            df.loc[idx, f'Yield last {x} rounds'] = rendimento
        else:
            df.loc[idx, f'Yield last {x} rounds'] = 0.0

        dicionario[mandante].insert(0, pt_mandante)
        if len(dicionario[mandante]) > x:
            dicionario[mandante].pop()

        dicionario[visitante].insert(0, pt_visitante)
        if len(dicionario[visitante]) > x:
            dicionario[visitante].pop()

    return df