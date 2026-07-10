"""
chatbot.py
==========

Chatbot de linha de comandos, 100% local, que responde SOMENTE com base
em documentos PDF colocados numa pasta "docs/". Ou seja, implementa RAG
(Retrieval-Augmented Generation):

  1. INDEXAÇÃO (feita uma vez, ou sempre que os PDFs mudam):
     - Lemos todos os PDFs da pasta "docs/".
     - Dividimos o texto de cada um em pedaços mais pequenos ("chunks").
     - Convertemos cada chunk num vetor numérico (embedding), usando um
       modelo de embeddings do Ollama.
     - Guardamos tudo (texto + embedding + origem) num ficheiro de cache
       local, para não termos de refazer este trabalho todas as vezes.

  2. CONVERSA (a cada pergunta do utilizador):
     - Convertemos a pergunta num embedding, da mesma forma.
     - Comparamos esse embedding com os de todos os chunks (usando
       similaridade de cosseno) para encontrar os pedaços de texto mais
       relacionados com a pergunta.
     - Construímos um prompt que inclui esses pedaços como "contexto" e
       pedimos ao modelo de chat (Gemma) para responder com base neles.

Pré-requisitos:
- Ollama instalado e a correr.
- Modelo de chat descarregado:      ollama pull gemma3
- Modelo de embeddings descarregado: ollama pull nomic-embed-text
- Dependências Python:               pip install -r requirements.txt
- Uma pasta "docs/" ao lado deste script, com ficheiros .pdf dentro.
"""

import os          # para percorrer a pasta docs/ e verificar ficheiros
import json        # para gravar/ler o cache do índice em disco
import hashlib     # para "assinar" o estado da pasta docs/ (deteção de mudanças)

import numpy as np           # para cálculos vetoriais (similaridade de cosseno)
from pypdf import PdfReader  # para extrair texto de ficheiros PDF
import ollama                 # para falar com o Ollama (chat + embeddings)


# --------------------------------------------------------------------------
# CONFIGURAÇÃO
# --------------------------------------------------------------------------

# Modelo usado para gerar as respostas (conversação normal).
CHAT_MODEL = "gemma4"

# Modelo usado para gerar embeddings (representações numéricas de texto).
# "nomic-embed-text" é um modelo pequeno e rápido, feito especificamente
# para tarefas de pesquisa/recuperação de texto (RAG).
EMBED_MODEL = "nomic-embed-text"

# Pasta onde estão os PDFs que servem de base de conhecimento.
DOCS_DIR = "docs"

# Ficheiro onde guardamos o índice já processado (chunks + embeddings),
# para não termos de reprocessar os PDFs a cada arranque do script.
CACHE_FILE = "index_cache.json"

# Tamanho de cada "chunk" de texto, em número de caracteres.
# Chunks demasiado grandes tornam a pesquisa menos precisa; chunks
# demasiado pequenos perdem contexto. 1000 é um valor equilibrado para
# texto corrido em português.
CHUNK_SIZE = 1000

# Sobreposição entre chunks consecutivos, em caracteres. Isto evita que
# uma frase importante fique "cortada ao meio" exatamente na fronteira
# entre dois chunks, perdendo-se informação.
CHUNK_OVERLAP = 150

# Quantos chunks (os mais relevantes) vamos passar ao modelo como
# contexto para cada pergunta.
TOP_K = 4

# Instrução base do chatbot. Reforça que ele só deve responder com base
# no contexto fornecido (retirado dos PDFs), evitando "inventar" respostas
# quando a informação não está nos documentos (fenómeno conhecido como
# alucinação).
SYSTEM_PROMPT = """Tu és um assistente que responde exclusivamente com base
nos documentos fornecidos como contexto. Regras importantes:
- Usa APENAS a informação presente no contexto para responderes.
- Se a resposta não estiver no contexto, diz claramente que não encontraste
  essa informação nos documentos disponíveis. Não inventes respostas.
- Sempre que possível, indica de que documento/página veio a informação.
- Responde em português europeu, de forma clara e objetiva."""

COMANDOS_SAIDA = ("sair", "exit", "quit")


# --------------------------------------------------------------------------
# INDEXAÇÃO DOS PDFs
# --------------------------------------------------------------------------

def calcular_fingerprint(docs_dir):
    """
    Cria uma "assinatura" (hash) que representa o estado atual da pasta
    docs/: quais ficheiros existem, o tamanho e a data de modificação
    de cada um.

    Serve para sabermos se o índice em cache ainda é válido, ou se algum
    PDF foi adicionado/alterado/removido desde a última vez que indexámos
    — nesse caso, reconstruímos o índice do zero.
    """
    ficheiros = sorted(os.listdir(docs_dir)) if os.path.isdir(docs_dir) else []
    partes = []
    for nome in ficheiros:
        if not nome.lower().endswith(".pdf"):
            continue
        caminho = os.path.join(docs_dir, nome)
        stats = os.stat(caminho)
        # Combinamos nome + tamanho + data de modificação numa string única.
        partes.append(f"{nome}:{stats.st_size}:{stats.st_mtime}")

    assinatura = "|".join(partes)
    # Convertemos a string num hash curto e fixo (MD5 é suficiente aqui,
    # já que não é usado para segurança, apenas para deteção de mudanças).
    return hashlib.md5(assinatura.encode("utf-8")).hexdigest()


def extrair_texto_dos_pdfs(docs_dir):
    """
    Lê todos os PDFs da pasta docs_dir e devolve uma lista de dicionários,
    um por página, no formato:
        {"texto": "...", "fonte": "nome_ficheiro.pdf", "pagina": 3}

    Guardamos a página de origem para depois podermos dizer ao utilizador
    de onde veio cada resposta.
    """
    paginas = []

    if not os.path.isdir(docs_dir):
        return paginas

    for nome_ficheiro in sorted(os.listdir(docs_dir)):
        if not nome_ficheiro.lower().endswith(".pdf"):
            continue  # ignoramos qualquer ficheiro que não seja PDF

        caminho = os.path.join(docs_dir, nome_ficheiro)
        try:
            leitor = PdfReader(caminho)
        except Exception as erro:
            print(f"[aviso] Não foi possível ler '{nome_ficheiro}': {erro}")
            continue

        for numero_pagina, pagina in enumerate(leitor.pages, start=1):
            texto = pagina.extract_text() or ""
            texto = texto.strip()
            if texto:  # ignoramos páginas vazias (ex.: páginas só com imagens)
                paginas.append({
                    "texto": texto,
                    "fonte": nome_ficheiro,
                    "pagina": numero_pagina,
                })

    return paginas


def dividir_em_chunks(texto, tamanho=CHUNK_SIZE, sobreposicao=CHUNK_OVERLAP):
    """
    Divide um bloco de texto grande em pedaços mais pequenos (chunks),
    com sobreposição entre eles.

    Exemplo simplificado com tamanho=10, sobreposicao=3:
        texto:   "ABCDEFGHIJKLMNOP"
        chunk 1: "ABCDEFGHIJ"
        chunk 2: "HIJKLMNOPQ"  (repete "HIJ" do chunk anterior)
        ...

    A sobreposição ajuda a não "cortar" ideias importantes exatamente na
    fronteira entre dois chunks.
    """
    chunks = []
    inicio = 0
    comprimento_total = len(texto)

    while inicio < comprimento_total:
        fim = inicio + tamanho
        chunks.append(texto[inicio:fim])
        # Avançamos o início do próximo chunk, recuando "sobreposicao"
        # caracteres para criar a sobreposição desejada.
        inicio += tamanho - sobreposicao

    return chunks


def gerar_embedding(texto):
    """
    Pede ao Ollama para converter um texto num vetor numérico (embedding),
    usando o modelo definido em EMBED_MODEL.

    Um embedding é uma lista de números (ex.: 768 valores) que representa
    o "significado" do texto num espaço matemático. Textos com significados
    parecidos ficam com vetores próximos entre si — é isso que nos permite
    depois procurar os chunks mais relevantes para uma pergunta.
    """
    resposta = ollama.embeddings(model=EMBED_MODEL, prompt=texto)
    return resposta["embedding"]


def construir_indice(docs_dir):
    """
    Constrói o índice completo: extrai texto dos PDFs, divide em chunks e
    gera um embedding para cada chunk.

    Devolve uma lista de dicionários no formato:
        {"texto": ..., "fonte": ..., "pagina": ..., "embedding": [...]}
    """
    paginas = extrair_texto_dos_pdfs(docs_dir)

    if not paginas:
        return []

    indice = []

    # Primeiro juntamos todos os chunks de todas as páginas, só para
    # podermos mostrar progresso ao utilizador (útil se houver muitos PDFs).
    todos_chunks = []
    for pagina in paginas:
        for chunk in dividir_em_chunks(pagina["texto"]):
            todos_chunks.append({
                "texto": chunk,
                "fonte": pagina["fonte"],
                "pagina": pagina["pagina"],
            })

    total_chunks = len(todos_chunks)
    print(f"A indexar {total_chunks} pedaços de texto retirados dos PDFs...")

    for i, chunk_info in enumerate(todos_chunks, start=1):
        embedding = gerar_embedding(chunk_info["texto"])
        indice.append({
            "texto": chunk_info["texto"],
            "fonte": chunk_info["fonte"],
            "pagina": chunk_info["pagina"],
            "embedding": embedding,
        })
        # Mostra progresso a cada 10 chunks, para não poluir o terminal.
        if i % 10 == 0 or i == total_chunks:
            print(f"  ... {i}/{total_chunks}")

    return indice


def carregar_ou_construir_indice(docs_dir):
    """
    Tenta carregar o índice a partir da cache em disco (CACHE_FILE).
    Se a cache não existir, ou se os PDFs tiverem mudado desde a última
    indexação (fingerprint diferente), reconstrói o índice do zero e
    grava a nova cache.
    """
    fingerprint_atual = calcular_fingerprint(docs_dir)

    # Tentamos ler a cache existente.
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            # Só reaproveitamos a cache se a "assinatura" da pasta docs/
            # for igual à que foi guardada da última vez.
            if cache.get("fingerprint") == fingerprint_atual:
                print("Índice carregado a partir da cache (index_cache.json).")
                return cache["indice"]
        except (json.JSONDecodeError, KeyError):
            # Cache corrompida ou em formato antigo — ignoramos e reconstruímos.
            pass

    # Se chegámos aqui, é preciso (re)construir o índice.
    print("A construir índice novo a partir dos PDFs em docs/ ...")
    indice = construir_indice(docs_dir)

    # Gravamos o resultado em cache, já com a fingerprint associada, para
    # a próxima vez que o script correr (se os PDFs não mudarem) podermos
    # simplesmente reutilizar este índice sem reprocessar tudo.
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint_atual, "indice": indice}, f)

    return indice


# --------------------------------------------------------------------------
# RECUPERAÇÃO (RETRIEVAL)
# --------------------------------------------------------------------------

def similaridade_cosseno(vetor_a, vetor_b):
    """
    Calcula a similaridade de cosseno entre dois vetores — uma medida de
    quão "parecidos" são em termos de direção (não de magnitude).

    O resultado varia entre -1 e 1:
        1  -> vetores praticamente idênticos em significado
        0  -> vetores sem relação
        -1 -> vetores com significados opostos

    Usamos isto para comparar o embedding da pergunta do utilizador com o
    embedding de cada chunk, e assim saber quais chunks são mais relevantes.
    """
    a = np.array(vetor_a)
    b = np.array(vetor_b)
    produto_escalar = np.dot(a, b)
    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b)

    if norma_a == 0 or norma_b == 0:
        return 0.0  # evita divisão por zero em casos-limite

    return produto_escalar / (norma_a * norma_b)


def recuperar_chunks_relevantes(pergunta, indice, top_k=TOP_K):
    """
    Dada uma pergunta e o índice completo de chunks, devolve os "top_k"
    chunks mais relevantes para essa pergunta.

    Passos:
    1. Convertemos a pergunta num embedding.
    2. Calculamos a similaridade dessa pergunta com CADA chunk do índice.
    3. Ordenamos os chunks pela similaridade (do mais para o menos parecido).
    4. Devolvemos apenas os top_k primeiros.
    """
    if not indice:
        return []

    embedding_pergunta = gerar_embedding(pergunta)

    # Para cada chunk, calculamos a sua similaridade com a pergunta e
    # guardamos o par (similaridade, chunk) para depois ordenar.
    resultados = []
    for chunk in indice:
        sim = similaridade_cosseno(embedding_pergunta, chunk["embedding"])
        resultados.append((sim, chunk))

    # Ordenamos do mais relevante (maior similaridade) para o menos
    # relevante, e ficamos só com os top_k primeiros.
    resultados.sort(key=lambda par: par[0], reverse=True)
    mais_relevantes = resultados[:top_k]

    return [chunk for _sim, chunk in mais_relevantes]


def formatar_contexto(chunks):
    """
    Junta os chunks recuperados num único bloco de texto, identificando a
    origem de cada um (ficheiro + página), para incluir no prompt enviado
    ao modelo de chat.
    """
    blocos = []
    for chunk in chunks:
        cabecalho = f"[Fonte: {chunk['fonte']}, página {chunk['pagina']}]"
        blocos.append(f"{cabecalho}\n{chunk['texto']}")

    return "\n\n---\n\n".join(blocos)


# --------------------------------------------------------------------------
# CICLO PRINCIPAL DO CHATBOT
# --------------------------------------------------------------------------

def main():
    print("Chatbot local com RAG (Ollama). Escreve 'sair' para terminar.\n")

    # Construímos (ou carregamos da cache) o índice de conhecimento a
    # partir dos PDFs em docs/. Isto acontece uma vez, no arranque.
    indice = carregar_ou_construir_indice(DOCS_DIR)

    if not indice:
        print(
            f"[aviso] Não foram encontrados PDFs em '{DOCS_DIR}/', ou não foi "
            "possível extrair texto deles. O chatbot vai continuar a funcionar, "
            "mas sem qualquer base de conhecimento — as respostas indicarão "
            "que não há contexto disponível.\n"
        )

    # Histórico "limpo" da conversa: aqui NÃO guardamos os pedaços de PDF
    # recuperados, apenas o system prompt e as perguntas/respostas reais.
    # Isto evita que o contexto enviado ao modelo cresça descontroladamente
    # à medida que a conversa avança (cada pergunta pode trazer contexto
    # totalmente diferente).
    historico = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        user_input = input("Você: ").strip()

        if user_input.lower() in COMANDOS_SAIDA:
            print("Até já!")
            break

        if not user_input:
            continue

        # 1. Recuperação: procuramos no índice os chunks mais relevantes
        #    para esta pergunta específica.
        chunks_relevantes = recuperar_chunks_relevantes(user_input, indice)

        if chunks_relevantes:
            contexto = formatar_contexto(chunks_relevantes)
        else:
            contexto = "(Nenhum contexto relevante foi encontrado nos documentos.)"

        # 2. Construímos a mensagem "aumentada" que vamos enviar ao modelo,
        #    combinando o contexto recuperado com a pergunta original.
        #    Esta mensagem NÃO é a que fica guardada no histórico (ver abaixo).
        mensagem_aumentada = (
            f"Contexto retirado dos documentos:\n{contexto}\n\n"
            f"Pergunta do utilizador: {user_input}"
        )

        # 3. Montamos a lista de mensagens para este pedido específico:
        #    o system prompt + todo o histórico de conversa "limpo" até
        #    agora + a mensagem aumentada com o contexto desta pergunta.
        mensagens_para_o_modelo = historico + [
            {"role": "user", "content": mensagem_aumentada}
        ]

        print("Bot: ", end="", flush=True)
        resposta_completa = ""

        for chunk in ollama.chat(
            model=CHAT_MODEL,
            messages=mensagens_para_o_modelo,
            stream=True,
        ):
            texto = chunk["message"]["content"]
            print(texto, end="", flush=True)
            resposta_completa += texto

        print("\n")

        # 4. Atualizamos o histórico "limpo": guardamos a pergunta ORIGINAL
        #    (sem o contexto injetado) e a resposta do modelo. Assim, o
        #    histórico mantém-se pequeno e legível, e cada pergunta futura
        #    recebe o seu próprio contexto fresco, relevante para ela.
        historico.append({"role": "user", "content": user_input})
        historico.append({"role": "assistant", "content": resposta_completa})


if __name__ == "__main__":
    main()
