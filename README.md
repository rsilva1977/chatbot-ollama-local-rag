Chatbot de Vendas Online com Ollama

Um chatbot 100% local em Python que responde a perguntas sobre vendas online, usando o Ollama e a técnica RAG (Retrieval-Augmented Generation). O chatbot baseia as suas respostas em documentos PDF armazenados numa pasta docs/.



✨ Funcionalidades





Respostas baseadas em documentos PDF (FAQs, manuais, registos de interações).



Indexação automática dos PDFs (feita uma vez ou sempre que os ficheiros mudam).



Conversa em streaming (respostas aparecem em tempo real).



Histórico de conversa mantido para contexto.



Sem dependência de APIs externas — tudo corre na tua máquina.



📋 Pré-requisitos





Python 3.8+



Ollama instalado e a correr localmente



Modelos do Ollama descarregados:

ollama pull gemma4           # Modelo de chat (para gerar respostas)
ollama pull nomic-embed-text # Modelo de embeddings (para indexar os PDFs)



🚀 Instalação





Clona o repositório (ou copia os ficheiros para a tua pasta):

 git clone https://github.com/rsilva1977/chatbot-ollama-local-rag.git
 cd chatbot-ollama-local-rag



Instala as dependências:

 pip install -r requirements.txt



Cria a pasta docs/ e adiciona os PDFs:





Coloca os ficheiros PDF com os dados de vendas online dentro da pasta docs/ (ex: faq-vendas-online.pdf, manuais-procedimentos-vendas.pdf, registos-interacoes-clientes.pdf).



▶️ Como usar





Garante que o Ollama está a correr (normalmente inicia automaticamente após a instalação).



Corre o script:

 python chatbot.py



Faz perguntas sobre vendas online, como:





"Como posso devolver um produto?"



"Quais são os métodos de pagamento disponíveis?"



"Qual é o prazo de entrega para o Modelo A?"



Para sair, escreve sair, exit ou quit.



⚙️ Configuração

No topo do ficheiro chatbot.py, podes ajustar as seguintes variáveis:

DOCS_DIR = "docs/"            # Pasta onde estão os PDFs
CHAT_MODEL = "gemma4"         # Modelo de chat do Ollama
EMBEDDING_MODEL = "nomic-embed-text"  # Modelo de embeddings do Ollama
CHUNK_SIZE = 500             # Tamanho máximo de cada chunk (em caracteres)
MAX_CONTEXT_CHUNKS = 3       # Número máximo de chunks a incluir no contexto
CACHE_FILE = "indice.json"   # Ficheiro onde é guardado o índice





DOCS_DIR: Caminho para a pasta com os PDFs.



CHAT_MODEL: Nome do modelo de chat instalado localmente (confirma com ollama list).



EMBEDDING_MODEL: Nome do modelo de embeddings instalado localmente.



CHUNK_SIZE: Tamanho máximo de cada pedaço de texto (chunk) extraído dos PDFs.



MAX_CONTEXT_CHUNKS: Número máximo de chunks a incluir no contexto para cada pergunta.



CACHE_FILE: Ficheiro onde é guardado o índice (embeddings + texto).



📁 Estrutura do projeto

.
├── chatbot.py         # Script principal do chatbot
├── requirements.txt   # Dependências do projeto
├── README.md          # Este ficheiro
└── docs/              # Pasta com os PDFs de base de conhecimento
    ├── faq-vendas-online.pdf
    ├── manuais-procedimentos-vendas.pdf
    └── registos-interacoes-clientes.pdf



🛠️ Tecnologias





Python — Linguagem de programação



Ollama — Para correr modelos LLM localmente



Gemma — Modelo de linguagem da Google (para chat)



Nomic Embed Text — Modelo de embeddings



PyPDF — Para extrair texto de PDFs



📄 Licença

Este projeto está disponível sob a licença MIT. Sente-te à vontade para usar, modificar e distribuir.



🤝 Contribuições

Contribuições, issues e sugestões são bem-vindas! Sente-te à vontade para abrir um pull request ou uma issue.



📞 Suporte

Para dúvidas ou problemas, contacta:





E-mail: rsilva@galeci.com



Projeto: Chatbot de Vendas Online

