### Realizado:
- Experimentos preliminares:
	- Modelos testados
		- OneClass SVM, s
		- Floresta de isomento.
	- Grid search com 3 hiper-parâmetros e poucos valores.
	- Agrupou os pontos em atividades
	- Respeitando a ordem temporal, mas treinou apenas na atividade anterior e aplicou na seguinte.
- Explorou alternativas à otimização bayesiana
	- http://hyperopt.github.io/hyperopt-sklearn/ (tem 5 algoritmos de busca)
- Artigos lidos:
	-  [https://netflixtechblog.com/machine-learning-for-fraud-detection-in-streaming-services-b0b4ef3be3f6](https://netflixtechblog.com/machine-learning-for-fraud-detection-in-streaming-services-b0b4ef3be3f6)
	- [https://arxiv.org/abs/2203.02124](https://arxiv.org/abs/2203.02124)

### Tarefas:
- Usar variaveis de ambiente para configurar o projeto (Checar uso do hydra).