# Ford - Teste Técnico - Processamento de Linguagem Natural (NLP)

**Objetivo**: Avaliar a habilidade dos candidatos em criar, treinar e preparar para deploy um modelo de classificação baseado em embeddings, lidando com um conjunto de dados com classes altamente desbalanceadas.

---

### Descrição do Problema

Você deve trabalhar com um conjunto de dados que contém embeddings pré-calculados (representações vetoriais de textos) e rótulos associados. O dataset possui classes desbalanceadas, com algumas categorias tendo pouquíssimos exemplos. O objetivo é desenvolver uma pipeline completa que:

- Realize o aumento (augmentation) dos dados para lidar com o desbalanceamento.
- Treine e avalie um modelo de classificação baseado nesses embeddings.
- Prepare o modelo para ser implantado (deploy).

### Instruções Gerais

Os candidatos devem organizar o projeto para ser funcional, bem estruturado e facilmente compreendido por outras pessoas. A qualidade do código, a modularidade, a clareza e a documentação serão criteriosamente avaliadas.

---

### Tarefas Obrigatórias
1. Análise Estatística Inicial

    Realize uma análise exploratória do conjunto de dados para entender:
    - O nível de desbalanceamento entre as classes.
    - A distribuição das dimensões dos embeddings.
    - Possíveis correlações entre os embeddings e as classes.
  
    Apresente visualizações (gráficos, tabelas) que facilitem o entendimento dos dados.

2. Pré-processamento dos Dados

    Detecte e trate valores ausentes ou inconsistentes nos embeddings ou nos rótulos.
    Normalize ou padronize os embeddings, caso necessário.
    Documente cada etapa do pré-processamento.

3. Estratégias para Data Augmentation

Implemente pelo menos duas estratégias de aumento de dados. Exemplos:

    SMOTE (Synthetic Minority Oversampling Technique): Para criar novos exemplos sintéticos das classes minoritárias.
    Geração de Dados Sintéticos com Ruído: Adicionar ruído aos embeddings existentes para criar variações.
    Técnicas Baseadas em Interpolação: Criar novos embeddings como combinações lineares de pontos existentes.
    Justifique suas escolhas e compare o impacto das técnicas no desempenho do modelo.

4. Engenharia de Features

    Explore possíveis transformações ou reduções de dimensionalidade (ex.: PCA, UMAP) para melhorar o desempenho e/ou eficiência do modelo.
    Analise a relevância das dimensões do embedding para as classes-alvo.

5. Benchmark de Modelos

    Treine pelo menos dois modelos de classificação diferentes (ex.: SVM, Random Forest, Redes Neurais Simples) usando os embeddings.
    Compare os resultados usando métricas como F1-Score, AUC-ROC e precisão para cada classe.
    Apresente uma análise crítica dos resultados e escolha um modelo para ser refinado e preparado para o deploy.

6. Preparação para Deploy

    Serializar o modelo final usando um formato como Pickle ou ONNX.
    Criar um script ou API simples para consumir o modelo treinado, permitindo a classificação de novos embeddings.
    Documentar o uso do modelo (ex.: requisitos de entrada, exemplos de chamada).

Critérios de Avaliação

    Qualidade do Código
        Organização, modularidade e legibilidade.
        Uso de boas práticas de programação (PEP8, nomes de variáveis claros, comentários relevantes).

    Estruturação do Projeto
        Estrutura de diretórios clara e bem organizada (ex.: src/, data/, notebooks/, tests/).
        Presença de scripts automatizados (ex.: Makefile ou requirements.txt).

    Documentação
        README claro explicando os objetivos do projeto, instruções de execução e principais decisões.
        Comentários no código detalhando lógica e escolhas.

    Soluções para o Desbalanceamento
        Criatividade e eficiência das estratégias aplicadas.
        Justificação baseada em evidências.

    Modelo Final
        Desempenho nas métricas definidas.
        Robustez e preparo para uso em produção.

Entrega

    Submeta o código via um repositório público no GitHub/GitLab.
    Inclua todos os arquivos necessários para rodar o projeto (exceto o dataset, se necessário).
    Prazo de entrega: 7 dias após o recebimento do teste.

Extras (Não Obrigatórios, Mas Diferenciais)

    Implementação de testes automatizados para validar funções principais.
    Uso de ferramentas como Docker para encapsular o projeto.
    Aplicação de técnicas mais avançadas de NLP ou aprendizado de máquina (ex.: modelos de linguagem pré-treinados).

Esse teste é desenhado para avaliar tanto habilidades técnicas quanto a capacidade de estruturar um projeto profissional. Boa sorte!
