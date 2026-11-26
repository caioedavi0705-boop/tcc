## Modelo BiLSTM

As redes recorrentes bidirecionais (BRNN) percorrem a sequência em sentido direto e inverso e fundem as saídas das duas passagens, de modo que, em cada posição temporal, ficam disponíveis informações do passado e do futuro \cite{artigoBrnn}. Em geral, essa combinação favorece o desempenho em relação a uma única rede unidirecional, pois amplia o contexto útil à predição—com o correspondente aumento do número de parâmetros livres do modelo. Embora a leitura reversa possa parecer incompatível com tarefas estritamente online e causais, há muitos problemas—como o reconhecimento de fala por segmentos—em que a saída só é necessária ao final do trecho; nesses casos, o atraso de uma passagem adicional é pequeno e o uso do contexto futuro aproxima o comportamento do modelo ao da interpretação humana de sequências.

Essa ideia bidirecional foi estendida às células LSTM, consolidando arquiteturas capazes de capturar dependências de longo alcance nos dois sentidos da série temporal \cite{artigoGraves}.

A arquitetura BiLSTM é uma extensão do conceito da LSTM e tem como princípio a utilização de duas redes independentes para o processamento dos dados de entrada, em que uma delas lê os dados na direção direta (da esquerda para direita), enquanto a outra lê os dados na direção reversa (da direita para esquerda) \cite{artigo9}. Um esquema do modelo BiLSTM é apresentado na Figura 3.  A maneira pela qual as saídas das redes com direções opostas são tratadas varia de algoritmo para algoritmo. Os \textit{outputs} podem ser somados, multiplicados, concatenados ou gerarem uma média que represente o \textit{output} final da BiLSTM. Dessa forma, pode-se obter uma melhor aprendizagem do modelo em função do tratamento simultâneo de informações passadas e futuras.
Nela, verificam-se duas camadas ocultas, onde as setas horizontais representam o fluxo de informações nas duas direções. Já as setas verticais e curvadas conectam a camada de entrada a uma camada oculta ou uma camada oculta à camada de saída.

## Modelo Transformer

O *Transformer*, proposto por Vaswani et al. (2017), é um modelo sequência-a-sequência formulado em termos de codificador (*encoder*) e decodificador (*decoder*) \cite{vaswani2017}. A entrada—no trabalho original, uma frase no idioma de origem—é primeiro transformada em vetores contínuos por uma camada de *embedding*; em seguida, o codificador produz representações da sequência inteira, e o decodificador gera a saída de forma autoregressiva (por exemplo, a tradução). A principal inovação consiste em substituir a recorrência de redes como RNN, LSTM e GRU por **autoatenção** (*self-attention*) e **codificação posicional**, de modo que todos os elementos da sequência possam ser processados em paralelo, preservando ainda a informação de ordem \cite{tutorialTransformers}. Esses mecanismos ampliam o alcance das dependências temporais sem exigir que a informação se propague passo a passo ao longo da cadeia recorrente, o que favorece o treinamento em hardware paralelo e aplicações em análise de séries temporais quando a entrada é um segmento completo de observações \cite{tutorialTransformers}.

A seguir descrevem-se os componentes internos do modelo; a montagem em pilhas de codificadores e decodificadores é detalhada ao final desta seção \cite{vaswani2017,tutorialTransformers}.

### Codificação posicional

A autoatenção, por si só, é equivariante à permutação dos instantes: trocar a ordem das amostras não altera o resultado se os vetores de entrada forem os mesmos. Para que o modelo distinga **onde** cada elemento se situa na sequência, soma-se a cada vetor de entrada um vetor de **codificação posicional** (PE), fixo e não treinável na formulação original \cite{vaswani2017,tutorialTransformers}.

Na versão **senoidal** de Vaswani et al., para a posição $pos$ no tempo e o índice de dimensão $i = 0, 1, \ldots, d/2 - 1$, com $d$ a dimensão do *embedding*:

$$
PE_{(pos,\,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right).
$$

Pares de dimensões adjacentes compartilham a mesma frequência; dimensões de menor índice variam mais rapidamente com $pos$, e dimensões de maior índice capturam escalas mais lentas ao longo da sequência. O resultado é somado elemento a elemento aos *embeddings*, de modo que sinal semântico (ou, em séries temporais, o vetor de características do instante) e índice temporal coexistam no mesmo espaço $d$-dimensional \cite{vaswani2017,tutorialTransformers}.

### Self-attention

O mecanismo de **autoatenção** estabelece, para cada posição da sequência, uma representação atualizada como combinação das demais posições. Intuitivamente, calcula-se uma medida de compatibilidade entre todos os pares de elementos e utiliza-se essa medida para formar uma **média ponderada** dos vetores de conteúdo \cite{tutorialTransformers}.

A partir do vetor de entrada $x_i$ na posição $i$, três projeções lineares aprendidas produzem **consulta** (*query*) $q_i$, **chave** (*key*) $k_i$ e **valor** (*value*) $v_i$:

$$
q_i = W_q x_i, \quad k_i = W_k x_i, \quad v_i = W_v x_i.
$$

O peso de atenção da posição $j$ ao atualizar a posição $i$ decorre do produto escalar entre $q_i$ e $k_j$, passado por *softmax* sobre $j$. A representação de saída em $i$ é a soma ponderada dos valores:

$$
z_i = \sum_{j=1}^{n} \mathrm{softmax}\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right) v_j,
$$

em que $d_k$ é a dimensão dos vetores de consulta e chave em cada cabeça. A divisão por $\sqrt{d_k}$ (atenção escalonada) evita que o *softmax* fique excessivamente concentrado quando $d_k$ cresce \cite{vaswani2017,tutorialTransformers}. Em notação matricial, com $Q$, $K$ e $V$ empilhando todas as posições:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$

Assim, cada instante pode, em uma única camada, incorporar evidência de **qualquer** outro instante da janela, com pesos aprendidos durante o treinamento.

### Atenção multi-cabeça (Multi-Head)

Um único mapa de atenção pode não bastar para capturar relações distintas entre os elementos (por exemplo, dependências de curto e de longo alcance). O **multi-head self-attention** executa, em paralelo, $h$ mecanismos de atenção com matrizes $W_q$, $W_k$ e $W_v$ **diferentes** em cada cabeça; cada cabeça projeta as entradas em um subespaço de dimensão menor $d_k$, de modo que $h \cdot d_k$ coincide com a dimensão de trabalho $d$ \cite{vaswani2017,tutorialTransformers}.

Para a cabeça $l$:

$$
\mathrm{head}_l = \mathrm{Attention}\!\left(Q W_q^{(l)},\, K W_k^{(l)},\, V W_v^{(l)}\right).
$$

As saídas das cabeças são **concatenadas** e linearmente projetadas de volta à dimensão $d$:

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)\, W_o.
$$

Entrada e saída da camada mantêm a mesma forma $(n \times d)$, o que permite empilhar blocos e empregar conexões residuais.

### Rede feed-forward e Conexão Residual

Cada bloco do codificador agrupa duas **subcamadas** transformadoras—atenção multi-cabeça e rede *feed-forward*—e, em ambas, o *Transformer* aplica o mesmo padrão de **conexão residual** seguido de **normalização por camada** (*layer normalization*) \cite{vaswani2017,tutorialTransformers}. Na conexão residual, a saída de uma subcamada $\mathrm{Sublayer}(\cdot)$ não substitui diretamente a entrada $x$; ela é **somada** a $x$, preservando um caminho pelo qual o gradiente e o sinal original podem atravessar o bloco com pouca distorção:

$$
x_{\mathrm{out}} = \mathrm{LayerNorm}\bigl(x + \mathrm{Sublayer}(x)\bigr).
$$

A **normalização por camada** é aplicada **depois** da soma residual. Para cada instante $i$ da sequência, toma-se o vetor $u_i \in \mathbb{R}^{d}$ resultante dessa soma e calculam-se a média $\mu_i$ e a variância $\sigma_i^2$ **ao longo das $d$ dimensões desse vetor** (e não ao longo do *batch* ou de todos os instantes). Em seguida, o vetor é reescalado, centralizado e passa por parâmetros aprendíveis de escala $\gamma$ e deslocamento $\beta$:

$$
\mathrm{LayerNorm}(u_i) = \gamma \odot \frac{u_i - \mu_i}{\sqrt{\sigma_i^2 + \varepsilon}} + \beta,
$$

em que $\odot$ denota o produto elemento a elemento e $\varepsilon$ é uma constante pequena para estabilidade numérica. Intuitivamente, cada posição temporal obtém representação com **média e variância controladas** nas suas $d$ coordenadas, o que reduz oscilações de magnitude entre blocos empilhados e melhora a convergência do treinamento em sequências \cite{vaswani2017,tutorialTransformers}. Os parâmetros $\gamma$ e $\beta$ permitem que a rede, após normalizar, recupere expressividade caso a escala uniforme não seja a mais adequada à tarefa.

Esse esquema de residual seguido de normalização, inspirado em redes residuais profundas, reduz o risco de degradação do treinamento quando se empilham vários blocos e facilita que cada subcamada aprenda **ajustes incrementais** em vez de remapear toda a representação do zero \cite{tutorialTransformers}. Como entrada e saída de atenção e de *feed-forward* têm a mesma dimensão $d$, a soma $x + \mathrm{Sublayer}(x)$ é dimensionalmente válida em todos os instantes da sequência.

No primeiro passo do bloco, $x$ percorre a atenção multi-cabeça e forma $x' = \mathrm{LayerNorm}\bigl(x + \mathrm{MultiHead}(x)\bigr)$. Em seguida, cada posição $i$ de $x'$ passa pela **rede feed-forward posicional** (*position-wise*): duas transformações lineares com ativação ReLU entre elas, com os **mesmos** pesos em todos os instantes \cite{vaswani2017,tutorialTransformers}:

$$
\mathrm{FFN}(x) = \max(0,\, x W_1 + b_1)\, W_2 + b_2.
$$

A rede atua de forma independente em cada instante, mas compartilha parâmetros ao longo da sequência; ela introduz não linearidade e amplia a capacidade representacional **depois** da mistura global promovida pela autoatenção. O bloco encerra com a segunda conexão residual:

$$
x_{\mathrm{bloco}} = \mathrm{LayerNorm}\bigl(x' + \mathrm{FFN}(x')\bigr),
$$

cuja saída alimenta o bloco seguinte na pilha \cite{vaswani2017,tutorialTransformers}.

### Montagem com codificadores e decodificadores

Na arquitetura completa, o *Transformer* combina **várias cópias** de dois tipos de módulo: **codificadores** (*encoders*) e **decodificadores** (*decoders*) \cite{tutorialTransformers}.

#### Bloco do codificador

Um bloco do codificador encadeia **atenção multi-cabeça** e **rede feed-forward**, cada subcamada envolvida por **conexão residual** e **normalização por camada** (*layer normalization*). As conexões residuais favorecem o treinamento de redes profundas e a estabilização do aprendizado; a normalização por camada é usual em modelos sequenciais e auxilia a convergência \cite{tutorialTransformers}. A saída de um bloco serve de entrada do bloco seguinte na pilha. O primeiro bloco do codificador recebe a sequência já representada como *embeddings* somados à **codificação posicional** \cite{tutorialTransformers}.

#### Bloco do decodificador

O bloco do decodificador repete o espírito do codificador, mas opera sobre **duas fontes de informação**: a saída do decodificador anterior e a representação produzida pelo **último bloco do codificador** \cite{tutorialTransformers}. Internamente, o decodificador dispõe de três subcamadas principais: (1) autoatenção multi-cabeça sobre a sequência alvo; (2) **atenção codificador–decodificador**, em que consultas provêm da autoatenção do decodificador e chaves e valores são extraídos da saída do codificador, ligando entrada e saída (por exemplo, idioma de origem e de destino); (3) rede feed-forward. Cada etapa é seguida de residual e normalização, como no codificador \cite{tutorialTransformers}.

#### Máscara na autoatenção do decodificador

Na autoatenção **dentro do decodificador**, aplica-se **máscara** sobre posições futuras da sequência alvo durante o treinamento. Assim, ao prever o elemento na posição $i$, o modelo não pode “ver” alvos nas posições $j > i$, respeitando a geração autoregressiva. No treino, em geral utiliza-se a sequência alvo correta (*teacher forcing*), e não a predição do passo anterior; na inferência, cada token gerado é incorporado ao decodificador (com *embedding* e PE) para produzir o seguinte \cite{tutorialTransformers}.

#### Empilhamento de blocos

O número de blocos empilhados no codificador e no decodificador é hiperparâmetro, de forma análoga às camadas ocultas em redes clássicas \cite{tutorialTransformers}. Uma diferença importante é que, ao longo da pilha, a dimensão das representações ($d$) **permanece constante**: não há redução de tamanho entre blocos sucessivos, ao contrário de algumas arquiteturas convolucionais \cite{tutorialTransformers}.

#### Saída do modelo

A saída do codificador superior é disponibilizada a **todos** os blocos do decodificador. Na extremidade da pilha, o vetor do decodificador pode passar por uma camada linear que projeta para o tamanho desejado—no exemplo de tradução automática, o vocabulário do idioma alvo—e, em seguida, por um *softmax* que estima a probabilidade do próximo símbolo \cite{tutorialTransformers}. Para **classificação** ou **regressão**, substituem-se essa camada linear final e a função de ativação de acordo com a tarefa (por exemplo, uma unidade linear para valor contínuo), podendo ainda dispensar-se o decodificador quando toda a sequência de entrada está disponível de uma vez—as abordagens *encoder-only* em séries temporais seguem essa linha \cite{tutorialTransformers}.