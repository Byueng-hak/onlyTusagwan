# AI 애플리케이션 개발

앞으로 AI 알고리즘은 점점 더 일상적인 애플리케이션에 통합 될 것입니다. 예를 들어 스마트 폰 앱에 이미지 분류기를 포함 할 수 있습니다. 이를 위해 전체 애플리케이션 아키텍처의 일부로 수십만 개의 이미지에 대해 학습 된 딥 러닝 모델을 사용합니다. 향후 소프트웨어 개발의 대부분은 이러한 유형의 모델을 응용 프로그램의 공통 부분으로 사용할 것입니다.

​    

이 프로젝트에서는 다양한 종류의 꽃을 인식하도록 이미지 분류기를 학습합니다. 카메라가보고있는 꽃의 이름을 알려주는 전화 앱에서 이와 같은 것을 사용하는 것을 상상할 수 있습니다. 실제로는이 분류기를 학습 한 다음 애플리케이션에서 사용하기 위해 내 보냅니다. 102 개의 꽃 카테고리로 구성된이 데이터 세트를 사용할 것입니다. 아래에서 몇 가지 예를 볼 수 있습니다.

<img src="/1.JPG" style="zoom: 50%;" />

프로젝트는 여러 단계로 나뉩니다.

​    

ㆍ 이미지 데이터 세트로드 및 전처리

ㆍ 데이터 세트에서 이미지 분류기 훈련

ㆍ 훈련 된 분류기를 사용하여 이미지 콘텐츠 예측

ㆍ Python으로 구현할 각 부분을 안내합니다.

​    이 프로젝트를 완료하면 레이블이 지정된 이미지 세트에 대해 학습 할 수있는 애플리케이션이 생깁니다. 여기에서 네트워크는 꽃에 대해 배우고 명령 줄 응용 프로그램으로 끝납니다. 그러나 새로운 기술로 수행하는 작업은 데이터 세트 구축에 대한 상상력과 노력에 달려 있습니다. 예를 들어 자동차 사진을 찍고 제조사와 모델이 무엇인지 알려주고 정보를 조회하는 앱을 상상해보십시오. 나만의 데이터 세트를 만들고 새로운 것을 만드세요.

​    먼저 필요한 패키지를 가져옵니다. 코드 시작 부분에 모든 가져 오기를 유지하는 것이 좋습니다. 이 노트북을 살펴보고 패키지를 가져와야한다는 것을 알게되면 여기에 가져 오기를 추가해야합니다.

​    CPU 모드가 아닌 GPU를 선택한 작업 공간에서이 노트북을 실행 중인지 확인하십시오.

In[1]:

```
# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
from collections import OrderedDict
import json
```

# Load the data

여기서`torchvision`을 사용하여 데이터를로드합니다 ([documentation] (http://pytorch.org/docs/0.3.0/torchvision/index.html)). 데이터는이 노트북과 함께 포함되어야합니다. 그렇지 않으면 [여기에서 다운로드] (https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) 할 수 있습니다. 데이터 세트는 교육, 검증 및 테스트의 세 부분으로 나뉩니다. 훈련을 위해 무작위 크기 조정, 자르기 및 뒤집기와 같은 변형을 적용하고 싶을 것입니다. 이렇게하면 네트워크가 일반화되어 더 나은 성능을 얻을 수 있습니다. 또한 사전 훈련 된 네트워크에서 요구하는대로 입력 데이터의 크기가 224x224 픽셀로 조정되었는지 확인해야합니다.

검증 및 테스트 세트는 아직 보지 못한 데이터에 대한 모델의 성능을 측정하는 데 사용됩니다. 이를 위해 크기 조정이나 회전 변형을 원하지는 않지만 크기를 조정 한 다음 이미지를 적절한 크기로 잘라야합니다.

사용할 사전 훈련 된 네트워크는 각 색상 채널이 개별적으로 정규화 된 ImageNet 데이터 세트에서 훈련되었습니다. 세 세트 모두에 대해 이미지의 평균과 표준 편차를 네트워크가 예상하는대로 정규화해야합니다. 평균은`[0.485, 0.456, 0.406]`이고 표준 편차는`[0.229, 0.224, 0.225]`이며 ImageNet 이미지에서 계산됩니다. 이 값은 각 색상 채널을 0에서 중심으로 이동하고 범위는 -1에서 1까지입니다.

In[2]:

```
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
using_gpu = torch.cuda.is_available()
```

In[3]:

``` 
# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
testval_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
image_testset = datasets.ImageFolder(test_dir, transform=testval_transforms)
image_valset = datasets.ImageFolder(valid_dir, transform=testval_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
image_trainloader = torch.utils.data.DataLoader(image_trainset, batch_size=64, shuffle=True)
image_testloader = torch.utils.data.DataLoader(image_testset, batch_size=64, shuffle=True)
image_valloader = torch.utils.data.DataLoader(image_valset, batch_size=64, shuffle=True)
```

# Label mapping

또한 범주 레이블에서 범주 이름으로의 매핑을로드해야합니다. cat_to_name.json 파일에서 찾을 수 있습니다. json 모듈로 읽을 수있는 JSON 객체입니다. 이렇게하면 정수로 인코딩 된 범주를 꽃의 실제 이름에 매핑하는 사전이 제공됩니다.

In[4]:

``` 
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

# Building and training the classifier

이제 데이터가 준비되었으므로 분류기를 구축하고 훈련 할 차례입니다. 평소와 같이 torchvision.models에서 사전 훈련 된 모델 중 하나를 사용하여 이미지 기능을 가져와야합니다. 이러한 기능을 사용하여 새로운 피드 포워드 분류기를 구축하고 훈련합니다.

우리는이 부분을 당신에게 맡길 것입니다. 이 섹션을 성공적으로 완료하기위한 지침은 루 브릭을 참조하십시오. 

해야 할 일 :

ㆍ사전 훈련 된 네트워크로드 (시작점이 필요한 경우 VGG 네트워크는 훌륭하게 작동하고 사용하기 쉽습니다)
ㆍReLU 활성화 및 드롭 아웃을 사용하여 훈련되지 않은 새로운 피드 포워드 네트워크를 분류기로 정의
ㆍ기능을 얻기 위해 사전 훈련 된 네트워크를 사용하여 역 전파를 사용하여 분류기 계층 훈련
ㆍ검증 세트에서 손실과 정확도를 추적하여 최상의 하이퍼 파라미터를 결정합니다.

아래에 셀을 열어 두었지만 필요한만큼 사용하십시오. 우리의 조언은 문제를 개별적으로 실행할 수있는 작은 부분으로 나누는 것입니다. 각 부분이 예상 한대로 작동하는지 확인한 후 다음으로 넘어갑니다. 각 부분을 진행하면서 돌아가서 이전 코드를 수정해야한다는 것을 알게 될 것입니다. 이것은 완전히 정상입니다!

훈련 할 때 피드 포워드 네트워크의 가중치 만 업데이트하고 있는지 확인하십시오. 모든 것을 올바르게 구축하면 70 % 이상의 유효성 검사 정확도를 얻을 수 있습니다. 최상의 모델을 찾으려면 다른 하이퍼 파라미터 (학습률, 분류기의 단위, 에포크 등)를 사용해보십시오. 프로젝트의 다음 부분에서 기본값으로 사용할 하이퍼 파라미터를 저장합니다.

작업 영역을 사용하여 코드를 실행하는 경우 마지막으로 중요한 팁 :이 노트북에서 장기 실행 작업 중에 작업 영역이 연결 해제되는 것을 방지하려면이 단원의 이전 페이지에서 세션 유지에 대한 GPU 작업 영역 소개를 읽어보십시오. 유효한. workspace_utils.py 모듈의 코드를 포함 할 수 있습니다.

In[5]:

```
# TODO: Build and train your network
epochs = 4
lr = 0.001
print_every = 10
```

In[6]:

``` 
# Freeze parameters so we don't backprop through them
hidden_layers = [10240, 1024]
def make_model(structure, hidden_layers, lr):
    if structure=="densenet161":
        model = models.densenet161(pretrained=True)
        input_size = 2208
    else:
        model = models.vgg16(pretrained=True)
        input_size = 25088
    output_size = 102
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('dropout',nn.Dropout(0.5)),
                              ('fc1', nn.Linear(input_size, hidden_layers[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(hidden_layers[1], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model

model = make_model('vgg16', hidden_layers, lr)
```

In[7]:

```
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
```

In[8]:

``` 
def cal_accuracy(model, dataloader):
    validation_loss = 0
    accuracy = 0
    for i, (inputs,labels) in enumerate(dataloader):
                optimizer.zero_grad()
                inputs, labels = inputs.to('cuda') , labels.to('cuda')
                model.to('cuda')
                with torch.no_grad():    
                    outputs = model.forward(inputs)
                    validation_loss = criterion(outputs,labels)
                    ps = torch.exp(outputs).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
    validation_loss = validation_loss / len(dataloader)
    accuracy = accuracy /len(dataloader)
    
    return validation_loss, accuracy
```

In[9]:

```
with active_session():
    def my_DLM(model, image_trainloader, image_valloader, epochs, print_every, criterion, optimizer, device='gpu'):
        epochs = epochs
        print_every = print_every
        steps = 0

        # change to cuda
        model.to('cuda')

        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(image_trainloader):
                steps += 1

                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    val_loss, train_ac = cal_accuracy(model, image_valloader)
                    print("Epoch: {}/{}... | ".format(e+1, epochs),
                          "Loss: {:.4f} | ".format(running_loss/print_every),
                          "Validation Loss {:.4f} | ".format(val_loss),
                          "Accuracy {:.4f}".format(train_ac))

                    running_loss = 0
    my_DLM(model, image_trainloader, image_valloader, epochs, print_every, criterion, optimizer, 'gpu')
```

Epoch: 1/4... |  Loss: 6.4524 |  Validation Loss 0.3465 |  Accuracy 0.0295

Epoch: 1/4... |  Loss: 4.4584 |  Validation Loss 0.3212 |  Accuracy 0.1000

Epoch: 1/4... |  Loss: 4.0043 |  Validation Loss 0.2779 |  Accuracy 0.2329

Epoch: 1/4... |  Loss: 3.4299 |  Validation Loss 0.2149 |  Accuracy 0.3459

Epoch: 1/4... |  Loss: 2.8110 |  Validation Loss 0.1520 |  Accuracy 0.4691

Epoch: 1/4... |  Loss: 2.3905 |  Validation Loss 0.1409 |  Accuracy 0.5067

Epoch: 1/4... |  Loss: 2.1229 |  Validation Loss 0.1574 |  Accuracy 0.5879

Epoch: 1/4... |  Loss: 1.8654 |  Validation Loss 0.0935 |  Accuracy 0.6126

Epoch: 1/4... |  Loss: 1.7904 |  Validation Loss 0.0804 |  Accuracy 0.6531

Epoch: 1/4... |  Loss: 1.7925 |  Validation Loss 0.0588 |  Accuracy 0.6823

Epoch: 2/4... |  Loss: 0.9521 |  Validation Loss 0.0989 |  Accuracy 0.6787 

​															.

​															.

Epoch: 4/4... |  Loss: 0.7457 |  Validation Loss 0.0560 |  Accuracy 0.8632

Epoch: 4/4... |  Loss: 0.7427 |  Validation Loss 0.0354 |  Accuracy 0.8579

# Test yout network

훈련 된 네트워크를 훈련이나 검증에서 본 적이없는 테스트 데이터, 이미지로 테스트하는 것이 좋습니다. 이렇게하면 완전히 새로운 이미지에서 모델의 성능에 대한 좋은 추정치를 얻을 수 있습니다. 네트워크를 통해 테스트 이미지를 실행하고 검증과 동일한 방식으로 정확도를 측정합니다. 모델이 잘 훈련 된 경우 테스트 세트에서 약 70 %의 정확도에 도달 할 수 있습니다.

In[10]:

```
# TODO: Do validation on the test set
def testing(dataloader):
    model.eval()
    model.to('cuda')
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in image_testloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _ , prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels.data).sum().item()
        print('Accuracy on the test set: %d %%' % (100 * correct / total))   
testing(image_testloader)
```

Accuracy on the test set: 85 %

# Save the checkpoint

이제 네트워크가 학습되었으므로 나중에 예측을 위해로드 할 수 있도록 모델을 저장합니다. 이미지 데이터 세트 중 하나 인 image_datasets [ 'train']. class_to_idx에서 가져온 인덱스에 대한 클래스 매핑과 같은 다른 항목을 저장하고 싶을 것입니다. 나중에 쉽게 추론 할 수있는 속성으로 모델에 연결할 수 있습니다.

model.class_to_idx = image_datasets['train'].class_to_idx

추론에 사용할 수 있도록 나중에 모델을 완전히 다시 빌드해야합니다. 체크 포인트에 필요한 모든 정보를 포함해야합니다. 모델을로드하고 훈련을 계속하려면 최적화 기 상태 인 optimizer.state_dict뿐만 아니라 epoch의 수를 저장해야합니다. 프로젝트의 다음 부분에서이 훈련 된 모델을 사용하고 싶을 것이므로 지금 저장하는 것이 가장 좋습니다.

In[11]:

```
model.class_to_idx = image_trainset.class_to_idx
```

In[11]:

```
# TODO: Save the checkpoint 

state = {
            'structure' :'vgg16',
            'learning_rate': lr,
            'epochs': epochs,
            'hidden_layers':hidden_layers,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx
}
torch.save(state, 'checkpoint.pth')
```

# Loading the checkpoint

이 시점에서 체크 포인트를로드하고 모델을 다시 빌드 할 수있는 함수를 작성하는 것이 좋습니다. 이렇게하면 네트워크를 재교육하지 않고도이 프로젝트로 돌아와 계속 작업 할 수 있습니다.

In[13]:

```
# TODO: Write a function that loads a checkpoint and rebuilds the model
def loading_checkpoint(path):
    
    # Loading the parameters
    state = torch.load(path)
    lr = state['learning_rate']
    structure = state['structure']
    hidden_layers = state['hidden_layers']
    epochs = state['epochs']
    
    # Building the model from checkpoints
    model = make_model(structure, hidden_layers, lr)
    class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])
    
loading_checkpoint('checkpoint.pth')
```

# Inference for classification

이제 추론을 위해 훈련 된 네트워크를 사용하는 함수를 작성합니다. 즉, 이미지를 네트워크로 전달하고 이미지에서 꽃의 종류를 예측합니다. 이미지와 모델을 취한 다음 확률과 함께 가장 가능성이 높은 $ K $ 클래스를 반환하는 predict라는 함수를 작성합니다. 다음과 같아야합니다.

```
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

먼저 네트워크에서 사용할 수 있도록 입력 이미지 처리를 처리해야합니다.

# Image Preprocessing

PIL을 사용하여 이미지를로드하는 것이 좋습니다 (문서). 모델의 입력으로 사용할 수 있도록 이미지를 전처리하는 함수를 작성하는 것이 가장 좋습니다. 이 함수는 훈련에 사용 된 것과 동일한 방식으로 이미지를 처리해야합니다.

먼저 가로 세로 비율을 유지하면서 가장 짧은면이 256 픽셀 인 이미지의 크기를 조정합니다. 이 작업은 축소판 또는 크기 조정 방법으로 수행 할 수 있습니다. 그런 다음 이미지의 중앙 224x224 부분을 잘라 내야합니다.

이미지의 색상 채널은 일반적으로 정수 0-255로 인코딩되지만 모델은 부동 소수점 0-1을 예상합니다. 값을 변환해야합니다. np_image = np.array (pil_image)와 같이 PIL 이미지에서 얻을 수있는 Numpy 배열을 사용하는 것이 가장 쉽습니다.

이전과 마찬가지로 네트워크는 이미지가 특정 방식으로 정규화 될 것으로 예상합니다. 평균의 경우 [0.485, 0.456, 0.406]이고 표준 편차의 경우 [0.229, 0.224, 0.225]입니다. 각 색상 채널에서 평균을 뺀 다음 표준 편차로 나눕니다.

마지막으로 PyTorch는 색상 채널이 첫 번째 차원이 될 것으로 예상하지만 PIL 이미지와 Numpy 배열에서 세 번째 차원입니다. ndarray.transpose를 사용하여 차원을 재정렬 할 수 있습니다. 색상 채널이 첫 번째 여야하고 다른 두 차원의 순서를 유지해야합니다.

In[14]:

```
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
   
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = image_transforms(pil_image)
    return img
```

In[15]:

```
# Demo
image_path = (test_dir + '/100/' + 'image_07939.jpg')
processed_image = process_image(image_path)
processed_image
```

Out[15]:

```
tensor([[[ 0.8789,  0.9132,  0.9474,  ...,  0.9474,  1.0673,  1.1529],
         [ 0.8789,  0.9132,  0.9474,  ...,  0.9988,  1.1358,  1.2557],
         [ 0.8789,  0.9132,  0.9303,  ...,  1.0502,  1.1872,  1.3070],
         ...,
         [-1.6727, -1.5870, -1.4500,  ..., -0.5596, -0.7650, -1.0390],
         [-1.6898, -1.6213, -1.5357,  ..., -0.3027, -0.5253, -0.8164],
         [-1.6727, -1.6555, -1.6213,  ..., -0.1657, -0.3541, -0.6109]],

        [[ 0.2927,  0.2752,  0.2577,  ...,  0.5728,  0.6954,  0.8179],
         [ 0.3102,  0.2927,  0.2927,  ...,  0.5553,  0.6779,  0.7829],
         [ 0.2927,  0.2752,  0.2752,  ...,  0.5028,  0.6429,  0.7654],
         ...,
         [-1.5980, -1.5630, -1.4755,  ..., -0.0049, -0.2850, -0.6001],
         [-1.6856, -1.6506, -1.5980,  ...,  0.1702, -0.0749, -0.3901],
         [-1.6331, -1.6331, -1.6506,  ...,  0.3102,  0.1001, -0.1800]],

        [[-0.5844, -0.5670, -0.5495,  ..., -0.2184, -0.0964,  0.0082],
         [-0.5495, -0.5321, -0.5147,  ..., -0.2184, -0.0964, -0.0092],
         [-0.5321, -0.5321, -0.5147,  ..., -0.2010, -0.0790,  0.0082],
         ...,
         [-1.4907, -1.4733, -1.4384,  ..., -0.6890, -0.9156, -1.1596],
         [-1.5256, -1.5256, -1.5081,  ..., -0.4973, -0.7587, -1.0550],
         [-1.4559, -1.4733, -1.5081,  ..., -0.3578, -0.5844, -0.8458]]])
```

작업을 확인하기 위해 아래 함수는 PyTorch 텐서를 변환하여 노트북에 표시합니다. process_image 함수가 작동하는 경우이 함수를 통해 출력을 실행하면 원본 이미지가 반환됩니다 (잘린 부분 제외).

In[16]:

```
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
```

In [17]:

```
image_path = (test_dir + '/100/' + 'image_07939.jpg')
imshow(processed_image.numpy())
```

Out[17]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fb4be6abf28>
```

<img src="./imgs/2.JPG" style="zoom: 50%;" />

# Class Prediction

이미지를 올바른 형식으로 얻을 수 있으면 모델로 예측을 수행하는 함수를 작성할 때입니다. 일반적인 관행은 가장 가능성이 높은 상위 5 개 정도 (일반적으로 top- $ K $라고 함)를 예측하는 것입니다. 클래스 확률을 계산 한 다음 $ K $ 가장 큰 값을 찾고 싶을 것입니다.

텐서에서 가장 큰 $ K $ 값을 얻으려면 x.topk (k)를 사용하십시오. 이 메서드는 가장 높은 k 개의 확률과 클래스에 해당하는 확률의 인덱스를 모두 반환합니다. 모델에 추가 한 class_to_idx를 사용하거나 데이터를로드하는 데 사용한 ImageFolder (여기 참조)를 사용하여 이러한 인덱스에서 실제 클래스 레이블로 변환해야합니다. 인덱스에서 클래스로의 매핑을 얻을 수 있도록 사전을 반전해야합니다.

다시 말하지만,이 메소드는 이미지와 모델 체크 포인트에 대한 경로를 취한 다음 확률과 클래스를 반환해야합니다.



```
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

In [62]:

```
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.cpu()
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        output = model.forward(img)
        probs, classes = torch.topk(input=output, k=topk)
        top_prob = probs.exp()

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in classes.cpu().numpy()[0]]
        
    print('Top Classes: ', top_classes)
    print('Top Probs: ', top_prob)
    return top_prob, top_classes
    
    
    #return top_prob.numpy()[0], mapped_classes
```

In [63]:

```
image_path = (test_dir + '/29/' + 'image_04095.jpg')
probs, classes = predict(image_path, model)

# Converting from tensor to numpy-array
print(probs)
print(classes)
Top Classes:  ['29', '14', '13', '10', '30']
Top Probs:  tensor([[ 0.7722,  0.1411,  0.0865,  0.0002,  0.0000]])
tensor([[ 0.7722,  0.1411,  0.0865,  0.0002,  0.0000]])
['29', '14', '13', '10', '30']
```

# Sanity Checking

이제 학습 된 모델을 예측에 사용할 수 있으므로 의미가 있는지 확인하십시오. 테스트 정확도가 높더라도 명백한 버그가 없는지 항상 확인하는 것이 좋습니다. matplotlib를 사용하여 상위 5 개 클래스에 대한 확률을 입력 이미지와 함께 막대 그래프로 플로팅합니다. 다음과 같이 표시되어야합니다.

<img src="./imgs/3.JPG" style="zoom: 50%;" />

cat_to_name.json 파일을 사용하여 클래스 정수 인코딩에서 실제 꽃 이름으로 변환 할 수 있습니다 (노트북에서 이전에로드되어야 함). PyTorch 텐서를 이미지로 표시하려면 위에 정의 된 imshow 함수를 사용하십시오.

In [64]:

```
# TODO: Display an image along with the top 5 classes
def sanity_checking():
    plt.rcParams["figure.figsize"] = (3,3)
    plt.rcParams.update({'font.size': 12})
    
    # Showing actual image
    image_path = (test_dir + '/29/' + 'image_04095.jpg')
    probs, classes = predict(image_path, model)
    #classes = classes.cpu().numpy()
    image_to_show = process_image(image_path)
    image = imshow(image_to_show.numpy(), ax = plt)
    image.axis('off')
    image.title(cat_to_name[str(classes[0])])
    image.show()
    
    # Showing Top Classes
    labels = []
    for class_index in classes:
        labels.append(cat_to_name[str(class_index)])
    y_pos = np.arange(len(labels))
    probs = probs[0]

    plt.barh(y_pos, probs, align='center', color='green')
    plt.yticks(y_pos, labels)
    plt.xlabel('Probability')
    plt.title('Top Classes')

    plt.show()

sanity_checking()
Top Classes:  ['29', '14', '13', '10', '30']
Top Probs:  tensor([[ 0.7722,  0.1411,  0.0865,  0.0002,  0.0000]])
```

<img src="./imgs/4.JPG" style="zoom: 50%;" />
