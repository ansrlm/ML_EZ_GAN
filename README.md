# ML_EZ_GAN
easy gan implementation with mnist data  
##### 개인 공부 목적으로 제작되었으며 어떤 상업적 의지도 없음을 미리 말씀드립니다.  
##### 저작권 관련 문제가 일어날 경우 바로 내리도록 하겠습니다.  
##### readme.md는 http://slazebni.cs.illinois.edu/spring17 를 기준으로 요약했습니다.  
#### GANs  
##### 최근 10년간 머신러닝 연구 중 가장 혁신적인 아이디어이며 2가지 특성을 가지고 있습니다.  
##### 1. Generative : Learn a generative model  
##### 2. Adversarial : Trained in an adversarial setting  
##### 이 두 특성에 대해서 이야기하고자 합니다.  
---
#### generative model  
##### GAN은 지도학습의 장점과 비지도학습의 장점 모두를 택하기 위해 준지도학습을 행합니다.  
##### 지도학습의 경우, discriminative model은 조건부확률 : p(Y|X)을 직접 구해서 클래스 간의 경계를 학습합니다. 이때 p(X)를 구하지 못하는 경우, 예를 들자면 특정 이미지를 볼 확률과 같은 경우는 p(X)를 통해서 sample을 할 수가 없습니다. 그래서 새로운 이미지를 만들 수 없는 상황에 봉착합니다.  
##### 이를 처리하기 위해 비지도학습으로 generative model를 선택하게 됩니다. generative model의 경우 p(X)를 model할 수 있고, 그래서 이를 통해서 새로운 이미지를 만들 수 있습니다.  
---
#### adversarial training  
##### 다음과 같은 순서로 GAN은 작동하게 됩니다.  
##### 1. generator : generator는 가짜 이미지를 생성하고, discriminator를 속이려 합니다.  
##### 2. discriminator : discriminator는 가짜 이미지와 진짜 이미지를 구별해서 처리하려 합니다.  
##### 3. 서로가 서로를 이기려고 학습합니다.  
##### 4. 계속 반복을 하고 이 반복에 따라서 generator와 discriminator는 더 좋은 성능을 가지게 됩니다.  
![ganimage1](./image/ganimage1.jpeg)  
![ganimage2](./image/ganimage2.jpeg)  
![ganimage3](./image/ganimage3.jpeg)  
---
#### GAN's formulation  
##### GAN은 minimax 게임처럼 작동하게 됩니다. (minimax game. 최소극대화 또는 미니맥스는 결정이론, 게임이론에서 사용하는 개념으로 최악의 경우 발생 가능한 손실(최대손실)을 최소화 한다는 규칙이다. wikipedia.)  
##### discriminator는 reward를 최대화 하려고 합니다. MAX V(D, G)  
##### generator는 이와 반대로 discriminator의 reward를 최소화 하려고 합니다. (discriminator의 loss를 최대화 하려고 합니다)  
![ganimage4](./image/ganimage4.jpeg)  
##### 위 식에 의거해 결국은 Nash 균형에 도달하게 됩니다. (Nash equilibrium. 내시 균형은 게임 이론에서 경쟁자 대응에 따라 최선의 선택을 하면 서로가 자신의 선택을 바꾸지 않는 균형상태를 말한다. 상대방이 현재 전략을 유지한다는 전제 하에 나 자신도 현재 전략을 바꿀 유인이 없는 상태를 말한다. wikipedia.)  
##### 유명한 비유법인 지폐위조범과 경찰의 예시에서, 지폐위조범은 generator로 가짜 지폐를 찍어냅니다. 경찰은 discriminator로 진짜 지폐를 가지고 가짜 지폐를 구분하려 합니다. 지폐위조범은 지폐를 사용하기 위해 더욱 진짜 지폐와 비슷해지도록 만들어내고, 경찰은 진짜 지폐와 구별하기 위해 더욱 정교한 방식으로 지폐를 판별하려고 합니다. 경찰은 결국 티끌 하나라도 다르면 바로 판별할 수 있는 방식을 얻어내지만, 지폐위조범도 또한 진짜 지폐와 같은 가짜 지폐를 만들게 됩니다.  
##### 이 최종적인 상황이 내시 균형에 놓은 상황이고 generator와 discriminator는 가장 높은 수준의 training을 가지게 됩니다.  
![ganimage5](./image/ganimage5.jpeg)  
##### D(X) = 1/2, 즉 진짜인지 가짜인지 판별할 수 없는 상황을 의미합니다.  
--- 
#### vanishing gradient strikes back  
![ganimage6](./image/ganimage6.jpeg)  
##### 두개의 네트워크가 트레이닝을 할때 discriminator가 너무 뛰어나면 반환되는 gradient값이 0이나 1에 매우 같아지게 됩니다. 그렇게 되면, generator가 gradient값이 제대로 반영되지 않아서 proceed가 일어나지 않습니다. 이와 반대로 generator가 너무 뛰어나면 discriminator가 real world sample을 fake로 판단할 확률이 증가합니다. 그래서 각자의 네트워크의 learning rates를 각각 설정해서 비슷한 학습 수준을 유지해야 합니다.  
---
#### GAN drawbacks?  
##### 1. GAN은 MNIST와 같이 단순한 이미지들은 잘 처리했으나, CIFAR과 같은 복잡한 이미지에 대해서는 좋은 이미지가 나오지 않았습니다.  
##### 2. learning rates를 적정수준을 설정해도 vanishing gradient 현상이 생기는 경우가 있습니다.  
##### 1을 해결하기 위해서 DCGAN이 나왔고, 2를 해결하기 위해서 LSGAN이 소개되었습니다.  
---
### 다음은 이 문제점들을 해결하려고 한 DCGAN, LSGAN을 정리해 보겠습니다.  
