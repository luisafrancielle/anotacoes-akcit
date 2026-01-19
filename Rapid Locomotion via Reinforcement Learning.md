# Rapid Locomotion via Reinforcement Learning 

### Por que só domain randomization costuma ser uma politica conservadora? 

Treinar uma única política para funcionar bem em todos os domínios aleatorizados (π_DR) até pode transferir, **mas tende a virar uma política conservadora**, porque não existe um mecanismo explícito de **adaptação ao domínio atual** (ex.: correr no gelo vs grama deveria induzir comportamentos diferentes).

A alternativa proposta: **fazer a política se especializar no domínio/dinâmica atual** usando _privileged information_ no treino e _adaptação online_ no teste.

### Teacher–Student + Informação Privilegiada (Asymmetric / Privileged Learning)

Informação privilegiada -> Eles definem um vetor de parâmetros dinâmicos **d_t** (propriedades do robô/terreno) usado no simulador: massa/payload, centro de massa, força do motor, atrito e restituição do chão etc. (amostrados em ranges).
    
- Isso é “privilegiado” porque **não dá para medir diretamente no robô real** durante a execução.

### Estrutura teacher–student (distillation/imit)

- **Teacher policy**: π_T(x_t, d_t) recebe o estado “normal” do robô **x_t** + os parâmetros privilegiados **d_t**.

- **Student policy**: π_S(x_t, x[t−h:t−1]) não vê d_t (parâmetros de domínio); em vez disso, usa **histórico de observações** para inferir o que precisa.

- Intuição: se o student consegue agir igual ao teacher sem ver d_t, então ele aprendeu **identificação de sistema online “implícita”**.

### Online adaptation

- O teacher não usa d_t diretamente: ele **comprime d_t num latente z_t** via um encoder g(d_t).
- A ação sai de um “policy body” comum: a_t = π_b(x_t, z_t).
- O student substitui o encoder por um módulo de adaptação h(x_hist) que estima **ẑ_t** a partir do histórico.

- Treino do student é um **alinhamento de representações**: minimizar (ẑ_t − z_t)², ou seja, fazer o latente estimado pelo student casar com o latente “ideal” do teacher.  

Obs.: Isso aproveita o policy body do teacher e treina “só” o módulo de adaptação.

### Por que isso pode ser mais simples/eficiente que outras alternativas?

- Eles usam um histórico curto (h = 15) “pequeno o suficiente” para rodar em tempo real junto do policy body.

- O teacher é otimizado com PPO; o student (módulo de adaptação) é otimizado com _supervised learning_ em dados on-policy.

### Ablation: “impacto do online system identification”

- Eles comparam: **teacher (privilegiado)** vs **student (sem d_t, com system ID)** vs **policy com domain randomization (sem adaptação)**, especialmente no regime de alta velocidade.

- Concluíram que o acesso a informação privilegiada melhora desempenho em todas as velocidades, e **o ganho é maior em alta velocidade**; e o **student quase iguala o teacher** usando só sensores/histórico.


- Em teste real (comando 6.0 m/s), a versão **com System ID** tem velocidade real média maior (≈3.81 m/s) do que **sem System ID** (≈2.49 m/s), e também melhora tracking em sim.  

- Eles observam um trade-off: **introduzir rugosidade/ruído de terreno no treino encolhe a “command area”** (faixa de comandos que o robô consegue rastrear) mesmo em terreno plano.  

Obs.: Avaliar se para o nosso objetivo seja melhor não tentar cobrir tudo com randomização pesada e sim randomizar o essencial (dinâmica relevante), e usar **adaptação online (teacher–student / system ID)** para especializar no domínio corrente.


### Informações da Simulação: 

Coletaram 400 milhões de timesteps simulados usando 4000 agentes em paralelo para o treinamento da política. Isso é aproximadamente equivalente a 92 dias em tempo real, que podemos simular em menos de três horas de tempo de relógio usando uma única GPU NVIDIA RTX 3090.
