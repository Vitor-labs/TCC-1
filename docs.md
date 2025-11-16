# Planejamento de 3 Meses - Modelo de Classificação de Failure Modes
## Projeto MLOps: Classificação Multiclasse de Comentários Técnicos em Failure Modes.

---

## **SPRINT 1 (Dias 1-15): Fundação e Coleta Inicial** - *FINALIZADO*

### **Objetivos Principais**
- Estabelecer infraestrutura base do projeto
- Iniciar coleta de dados
- Elicitar requisitos críticos de negócio

### **Atividades Detalhadas**

#### **Trilha 1: Setup e Coleta de Dados** (70% do esforço) - *FEITO*
- **Dias 1-3:**
  - Configurar repositório Git e estrutura de pastas do projeto ✔
  - Definir ambiente de desenvolvimento (Python, bibliotecas, versionamento) ✔
  - Mapear fontes de dados disponíveis (sistemas legados, bancos de dados, planilhas) ✔
  - Documentar dicionário de dados preliminar ✔
  	>	nome da coluna (tipo): descrição.

- **Dias 4-10:**
  - Extrair dados históricos de comentários técnicos ✔
  - Coletar labels existentes de Failure Modes (se disponíveis) - *Postergado*
	> Precisa adicionar failure modes na coleta semanal
  - Implementar scripts de extração automatizada ✔

- **Dias 11-15:**
  - Validar qualidade inicial dos dados coletados ✔
  - Identificar problemas de encoding, formatos, missing values ✔
  - Criar primeira versão do dataset bruto (versionamento DVC/MLflow) - *Postergado*
  	> Criar DVC raw (pesquisa do gsar) e DVC interim (processado). 

#### **Trilha 2: Elicitação de Requisitos** (30% do esforço - **PARALELA**) - *FEITO*
- **Dias 1-5:**
  - Reuniões com stakeholders (engenheiros, qualidade, service) ✔
  - Documentar categorias de Failure Modes existentes ✔

- **Dias 6-12:**
  - Definir escopo das classes (quantas? hierarquia?) ✔
  - Estabelecer prioridades de negócio ✔
  - Definir métricas de sucesso do projeto (accuracy mínima, F1-score por classe)
	> F1 + AUC - PR + MCC (não decisivo) + AUC - ROC. -> Front de Parreto.

- **Dias 13-15:**
  - Documentar requisitos funcionais e não-funcionais ✔
  - Criar documento de especificação v1.0 - *Postergado*

### **Entregáveis da Sprint 1**
✅ Dataset bruto versionado (mínimo 1000-5000 amostras)  
✅ Documento de requisitos de negócio v1.0  
✅ Taxonomia de Failure Modes definida  
✅ Infraestrutura base configurada  

### **SOBRAS**
- Adicionar compo verdade "failure mode" na pesquisa do GSAR.
- Versionar o dataset com DVC + Data Dict.

---

## **SPRINT 2 (Dias 16-30): Análise Exploratória e Refinamento de Requisitos**

### **Objetivos Principais**
- Compreender profundamente os dados
- Finalizar definições de entrada/saída
- Identificar desafios técnicos

### **Atividades Detalhadas**

#### **Trilha 1: EDA - Análise Exploratória** (60% do esforço)
- **Dias 16-20:**
  - Análise estatística descritiva (distribuição de classes, comprimento de textos) ✔
  - Visualizações (word clouds por classe, distribuição temporal) ✔
  - Análise de desbalanceamento de classes ✔
  - Identificação de outliers e anomalias - *Postergado*

- **Dias 21-25:**
  - Análise de qualidade textual (erros de digitação, abreviações, jargões) ✔
  - Estudo de correlações entre features (se houver metadados) - *Em progresso*
  - Análise de vocabulário específico por Failure Mode - *Em progresso*
  - Identificação de padrões linguísticos - *Em progresso*

- **Dias 26-30:**
  - Documentar insights principais em relatório técnico (Lib do leo para correlação)
  - Identificar necessidades de augmentation/balanceamento - *Descartado*
  - Criar notebook de EDA versionado - *Em progresso*

#### **Trilha 2: Refinamento de Requisitos** (40% do esforço - **PARALELA**)
- **Dias 16-22:**
  - Definir formato de entrada (texto puro, campos estruturados, metadados) ✔
  - Definir formato de saída (classe única, probabilidades, top-K) ✔
  - Estabelecer regras de negócio (thresholds de confiança, casos edge) ✔

- **Dias 23-30:**
  - Ajustar/mesclar classes se necessário - *Em progresso*
  - Definir pipeline de feedback dos usuários - *Em progresso*

### **Entregáveis da Sprint 2**
✅ Relatório de EDA completo  
✅ Notebooks analíticos versionados  
✅ Documento de requisitos v2.0 (refinado)  
✅ Definição final de entradas/saídas  
✅ Plano de tratamento de dados

---

## **SPRINT 3 (Dias 31-45): Preparação de Dados e Estratégia de Modelagem**

### **Objetivos Principais**
- Preparar dados para treinamento
- Definir estratégia de features e modelos
- Estabelecer baseline

### **Atividades Detalhadas**

#### **Dias 31-35: Preparação de Dados**
- Implementar pipeline de pré-processamento:
  - Limpeza de texto (remoção de ruído, normalização)
  - Tratamento de abreviações automotivas
  - Tokenização e normalização
  - Handling de missing values
- Estratégias de balanceamento de classes:
  - Análise de SMOTE, oversampling, undersampling
  - Data augmentation (parafrasear, back-translation)
- Split estratificado (train/validation/test - 70/15/15 ou 80/10/10)
- Versionamento de datasets processados

#### **Dias 36-40: Engenharia de Features**
- **Features Textuais:**
  - TF-IDF (diferentes configurações de n-gramas)
  - Word embeddings (Word2Vec, GloVe)
  - Embeddings contextualizados (BERT, RoBERTa)
  - Features específicas do domínio automotivo

- **Features Adicionais (se aplicável):**
  - Metadados (timestamp, fonte, localização)
  - Features estatísticas (comprimento, complexidade)
  - Features de sentimento

- **Documentação:**
  - Feature store inicial
  - Registro de transformações aplicadas

#### **Dias 41-45: Levantamento de Modelos**
- **Modelos Candidatos:**
  - **Baseline Simples:** Naive Bayes, Logistic Regression
  - **Modelos Tradicionais:** Random Forest, XGBoost, SVM
  - **Deep Learning:** CNN para texto, LSTM/GRU, BiLSTM
  - **Transformers:** BERT-based (BERT, DistilBERT, RoBERTa)
  - **Modelos específicos:** Automotive-domain fine-tuned models (se disponíveis)

- **Definir Métricas de Avaliação:**
  - Accuracy global
  - F1-score macro/micro/weighted
  - Precision/Recall por classe
  - Confusion matrix
  - ROC-AUC (one-vs-rest)
  - Métricas de negócio customizadas

- **Setup de Experimentação:**
  - Configurar MLflow/Weights & Biases
  - Definir protocolo de experimentação
  - Estabelecer ambiente de treinamento (GPU/CPU)

### **Entregáveis da Sprint 3**
✅ Pipeline de pré-processamento versionado  
✅ Datasets tratados e versionados  
✅ Catálogo de features documentado  
✅ Lista de modelos candidatos priorizada  
✅ Framework de experimentação configurado  
✅ Definição de métricas e critérios de sucesso  

---

## **SPRINT 4 (Dias 46-60): Treinamento Iterativo e Seleção de Modelos**

### **Objetivos Principais**
- Executar rodadas de treinamento
- Comparar performance de modelos
- Selecionar melhores candidatos

### **Atividades Detalhadas**

#### **Dias 46-50: Rodada 1 - Modelos Baseline**
- Treinar modelos simples (Naive Bayes, Logistic Regression, Random Forest)
- Avaliar performance no conjunto de validação
- Análise de erros:
  - Quais classes são mais confundidas?
  - Onde o modelo falha mais?
  - Padrões nos erros
- Documentar resultados no MLflow
- Estabelecer baseline de performance

#### **Dias 51-55: Rodada 2 - Modelos Avançados**
- Treinar modelos de ensemble (XGBoost, LightGBM)
- Experimentar com redes neurais (CNN, LSTM)
- Testar diferentes configurações de features
- Comparar performance vs baseline
- Análise de trade-offs (performance vs complexidade vs latência)

#### **Dias 56-60: Rodada 3 - Transformers e Fine-tuning**
- Fine-tuning de modelos BERT-based
- Testar variantes (DistilBERT para eficiência, RoBERTa para performance)
- Experimentar com domain adaptation (se houver dados automotivos pré-treinados)
- Ensemble de modelos (stacking, voting)
- Seleção dos top 2-3 modelos candidatos
- Análise comparativa detalhada

#### **Atividades Contínuas:**
- Registrar todos os experimentos (hiperparâmetros, métricas, artifacts)
- Versionar modelos treinados
- Documentar insights de cada rodada
- Reuniões de checkpoint (mid-sprint e end-sprint)

### **Entregáveis da Sprint 4**
✅ Mínimo 15-20 experimentos registrados  
✅ Top 2-3 modelos candidatos selecionados  
✅ Relatório comparativo de performance  
✅ Análise de erros detalhada  
✅ Modelos versionados e armazenados  

---

## **SPRINT 5 (Dias 61-75): Otimização e Preparação para Deploy**

### **Objetivos Principais**
- Otimizar modelos selecionados
- Preparar infraestrutura de deployment
- Validar performance final

### **Atividades Detalhadas**

#### **Dias 61-66: Otimização de Hiperparâmetros**
- **Otimização Automatizada:**
  - Grid Search / Random Search para modelos tradicionais
  - Bayesian Optimization (Optuna, Hyperopt)
  - Auto-tuning para transformers (learning rate, batch size, epochs)

- **Otimização de Dados:**
  - Refinar estratégias de balanceamento
  - Testar diferentes combinações de features
  - Limpeza adicional baseada em análise de erros
  - Data augmentation refinado

- **Otimização de Métricas:**
  - Ajustar thresholds de classificação
  - Calibração de probabilidades (se necessário)
  - Weighted loss functions para classes críticas

#### **Dias 67-70: Validação Final**
- Avaliar modelos otimizados no conjunto de teste (primeira vez)
- Validação cruzada estratificada
- Testes de robustez:
  - Performance em diferentes períodos temporais
  - Performance por subfonte de dados
  - Testes de adversarial examples
- Validação com stakeholders (amostra de predições)
- Selecionar modelo final para produção

#### **Dias 71-75: Preparação para Deploy**
- **Containerização:**
  - Criar Dockerfile do modelo
  - Otimizar tamanho da imagem
  - Testar localmente

- **API/Service:**
  - Desenvolver API REST (FastAPI/Flask)
  - Implementar endpoints de predição
  - Health checks e monitoring endpoints
  - Documentação de API (Swagger/OpenAPI)

- **Infraestrutura:**
  - Definir arquitetura de deployment (cloud, on-premise)
  - Setup de CI/CD pipeline (GitHub Actions, Jenkins, GitLab CI)
  - Configurar ambientes (dev, staging, prod)

### **Entregáveis da Sprint 5**
✅ Modelo otimizado final selecionado  
✅ Relatório de validação final  
✅ API do modelo desenvolvida e testada  
✅ Container Docker funcional  
✅ Pipeline CI/CD configurado  
✅ Documentação técnica de deployment  

---

## **SPRINT 6 (Dias 76-90): Integração, Deploy e Monitoramento**

### **Objetivos Principais**
- Deployar modelo em produção
- Implementar sistema de monitoramento
- Estabelecer processo de refinamento contínuo

### **Atividades Detalhadas**

#### **Dias 76-80: Deploy em Produção**
- **Staging:**
  - Deploy em ambiente de staging
  - Testes de integração com sistemas existentes
  - Testes de carga e performance (latência, throughput)
  - Validação com usuários beta

- **Produção:**
  - Deploy gradual (canary/blue-green deployment)
  - Configurar load balancing e auto-scaling
  - Implementar rollback automatizado
  - Documentar runbook operacional

- **Integrações:**
  - Conectar com sistemas de entrada de dados
  - Integrar com dashboards/ferramentas de usuários
  - Setup de logging centralizado

#### **Dias 81-85: Sistema de Monitoramento**
- **Monitoring de Performance:**
  - Métricas de modelo (accuracy, drift detection)
  - Métricas de sistema (latência, uptime, CPU/memória)
  - Dashboards de monitoramento (Grafana, Kibana)
  - Alertas automatizados

- **Data Drift Detection:**
  - Implementar detecção de drift de entrada (feature drift)
  - Monitorar distribuição de predições
  - Alertas de degradação de performance
  - Trigger para retreinamento

- **Feedback Loop:**
  - Sistema de coleta de feedback de usuários
  - Captura de predições incorretas
  - Armazenamento para retreinamento futuro

#### **Dias 86-90: Versionamento e Refinamento**
- **Versionamento:**
  - Documentar versão v1.0 do modelo
  - Registro completo de artifacts (código, dados, modelo)
  - Documentação de API e uso
  - Criar changelog

- **Processo de Refinamento Contínuo:**
  - Estabelecer cadência de retreinamento (mensal, trimestral)
  - Definir triggers para retreinamento automático
  - Processo de incorporação de novos dados
  - Pipeline de A/B testing para novos modelos

- **Documentação e Treinamento:**
  - Manual de usuário
  - Documentação técnica completa
  - Sessões de treinamento com usuários
  - Knowledge transfer para equipe de operações

- **Retrospectiva e Planejamento:**
  - Retrospectiva do projeto
  - Lições aprendidas
  - Roadmap de melhorias (v2.0)
  - Plano de manutenção

### **Entregáveis da Sprint 6**
✅ Modelo em produção (v1.0)  
✅ Sistema de monitoramento ativo  
✅ Dashboards de métricas  
✅ Pipeline de feedback implementado  
✅ Sistema de detecção de drift configurado  
✅ Documentação completa (técnica e usuário)  
✅ Processo de refinamento estabelecido  
✅ Plano de manutenção e evolução  

---

## **ARQUITETURA MLOps - Visão Geral**

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
│  (Service Records, Warranty Claims, Technical Reports, etc.)    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATA PIPELINE                                  │
│  ┌──────────┐    ┌──────────┐    ┌─────────────┐               │
│  │ Ingestion│───▶│Processing│───▶│ Versioning  │               │
│  │          │    │          │    │   (DVC)     │               │
│  └──────────┘    └──────────┘    └─────────────┘               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                               │
│  ┌──────────┐    ┌──────────┐    ┌─────────────┐               │
│  │   EDA    │───▶│ Feature  │───▶│   Training  │               │
│  │          │    │   Eng    │    │             │               │
│  └──────────┘    └──────────┘    └──────┬──────┘               │
│                                          │                       │
│  ┌──────────────────────────────────────▼──────────────────┐   │
│  │        Experiment Tracking (MLflow/W&B)                  │   │
│  └──────────────────────────────────────┬──────────────────┘   │
└─────────────────────────────────────────┼───────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL REGISTRY                                │
│             (Versioned Models + Metadata)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT PIPELINE                             │
│  ┌──────────┐    ┌──────────┐    ┌─────────────┐               │
│  │   Build  │───▶│   Test   │───▶│    Deploy   │               │
│  │ (Docker) │    │ (CI/CD)  │    │  (K8s/ECS)  │               │
│  └──────────┘    └──────────┘    └─────────────┘               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SERVING LAYER                                  │
│            ┌─────────────────────────────┐                      │
│            │     REST API (FastAPI)      │                      │
│            └────────────┬────────────────┘                      │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MONITORING & FEEDBACK                           │
│  ┌──────────┐    ┌──────────┐    ┌─────────────┐               │
│  │Performance│   │   Data   │    │   Feedback  │               │
│  │ Monitoring│   │   Drift  │    │  Collection │               │
│  └─────┬────┘    └────┬─────┘    └──────┬──────┘               │
│        └──────────────┼─────────────────┘                       │
│                       ▼                                          │
│            ┌──────────────────┐                                 │
│            │ Retraining       │                                 │
│            │ Trigger          │                                 │
│            └──────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## **STACK TECNOLÓGICO RECOMENDADO**

### **Data & Processing**
- **Storage:** AWS S3 / Azure Blob Storage / Google Cloud Storage
- **Processing:** Pandas, PySpark (se grande volume)
- **Versionamento:** DVC (Data Version Control)

### **ML & Experimentation**
- **Framework:** PyTorch / TensorFlow / Scikit-learn
- **Transformers:** HuggingFace Transformers
- **Experiment Tracking:** MLflow / Weights & Biases
- **Hyperparameter Tuning:** Optuna / Ray Tune

### **Deployment & Serving**
- **API:** FastAPI / Flask
- **Containerização:** Docker
- **Orquestração:** Kubernetes / AWS ECS / Azure AKS
- **CI/CD:** GitHub Actions / GitLab CI / Jenkins

### **Monitoring & Observability**
- **Logs:** ELK Stack (Elasticsearch, Logstash, Kibana)
- **Metrics:** Prometheus + Grafana
- **Drift Detection:** Evidently AI / Alibi Detect
- **APM:** DataDog / New Relic

---

## **MÉTRICAS DE SUCESSO DO PROJETO**

### **Métricas Técnicas**
- **Accuracy Global:** ≥ 85%
- **F1-Score Macro:** ≥ 0.80
- **F1-Score por Classe Crítica:** ≥ 0.75
- **Latência de Predição:** < 200ms (p95)
- **Disponibilidade:** ≥ 99.5%

### **Métricas de Negócio**
- **Redução de Tempo de Classificação Manual:** ≥ 60%
- **Taxa de Aceitação de Predições:** ≥ 80%
- **Cobertura de Failure Modes:** 100% das categorias principais
- **ROI:** Positivo em 6 meses

### **Métricas de Qualidade**
- **Cobertura de Testes:** ≥ 80%
- **Documentação:** 100% de APIs e processos críticos
- **Tempo de Retreinamento:** < 4 horas

---

## **RISCOS E MITIGAÇÕES**

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Dados insuficientes ou de baixa qualidade | Alta | Alto | Iniciar data augmentation cedo; envolver SMEs para rotulação; considerar active learning |
| Desbalanceamento severo de classes | Média | Alto | Técnicas de balanceamento (SMOTE, class weights); métricas adequadas (F1, não apenas accuracy) |
| Latência alta em produção | Média | Médio | Testar modelos leves (DistilBERT); otimização de inferência; caching |
| Drift de dados após deploy | Alta | Alto | Monitoring robusto; retreinamento automatizado; alertas configurados |
| Resistência dos usuários | Média | Médio | Envolvimento desde Sprint 1; treinamento; feedback loop ativo |
| Atraso em integrações de sistemas | Média | Médio | Começar conversas técnicas cedo; APIs desacopladas; mocks para desenvolvimento |

---

### **Checkpoints**
- **Semana 2:** Status de dados e requisitos
- **Semana 4:** Resultados de EDA e estratégia
- **Semana 6:** Primeiros resultados de modelos
- **Semana 8:** Modelo candidato selecionado
- **Semana 10:** Go/No-go para produção
- **Semana 12:** Revisão pós-deploy

### **Documentação Contínua**
- **README atualizado** a cada sprint
- **Architecture Decision Records (ADRs)** para decisões importantes
- **Runbooks** desenvolvidos desde Sprint 5
- **Model Cards** documentando características do modelo
