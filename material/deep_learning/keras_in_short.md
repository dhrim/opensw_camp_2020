
# TessorFlow

Google에서 개발한 Deep Larning Framework.

기본적인 알고리즘과 GPU, 분산 계산 처리를 해준다.

<br>


# Keras

TensorFlow를 백엔드로 하는 추상 프레임웤.

TensorFlow에 포함되어 있다.

- Reference : https://www.tensorflow.org/api_docs/python/tf/keras

<br>


# Keras 딥러닝 코드 Template

Keras 딥러닝 코드는 다음의 코드 구조를 갖는다.

```
# 데이터 준비
x = ...
y = ...

# 모델 구조 정의
model = keras.Sequential( ... )

# 모델 학습 방법 정의
model.compile(optimizer=..., loss=..., ...)

# 학습
model.fit(x, y, ...)

# 평가
loss, acc = model.evaluate(test_input, test_output)

# 사용
prediction = model.predict(test_input)
```

<br>


# 모델 구조 정의

DNN은 앞의 레이어의 결과를 뒤의 레이어의 입력으로 사용하는 방식으로 구성된다. 다양한 레이어들이 있고, 이를 추가하여 DNN의 구조를 정의한다.

<br>


2가지 방법

Sequential을 생성하면서 정의

```
from tensorflow.keras import Sequential

model = Sequential([
    Dense(4, activation='relu', input_shape(5,)),
    Dense(4, activation='relu'),
    Dense(1)
])
```

Sequential을 정의 후 add()호출하여

```
from tensorflow.keras import Sequential

model = Sequential()
model.add(Dense(4, activation='relu', input_shape(5,)))
model.add(Dense(32, input_shape=(500,)))
model.add(Dense(32))
```

referece

- Sequential : https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
- layer : https://www.tensorflow.org/api_docs/python/tf/keras/layers

<br>


# layer 들

모델을 구성할 때 포함할 수 있는 레이어들

https://www.tensorflow.org/api_docs/python/tf/keras/layers

- Dense
- Conv2D : CNN
- MaxPool2D : CNN
- Flatten : 2D, 3D 데이터를 Dense 1D로.
- DropOut : overfitting 처리
- BatchNormalization : overfitting 처리

<br>


# 레이어 Dense()

일반적인 레이어이다.

앞의 레이어의 전체 노드와 full connection을 갖는다고 해서 이름이 Dense이다.

https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

```
from tensorflow.keras.layers import Dense

Dense(
    32, # 노드 수
    activation=..., # 활성화 함수 종류
    kernel_initializer=..., # 웨이트 쵝화 방법
    bias_initializer=..., # bias 초기화 방법
    kernel_regularizer=..., # 웨이트 정규화 방법. overfitting 방지 위한
    bias_regularizer=..., # bias 정규화 방법. overfitting 방지 위한
)
```

```
Dense(32, input_shape=(16,))

Dense(32)
```

<br>


## 파라미터 activation

https://www.tensorflow.org/api_docs/python/tf/keras/activations

- 'relu'

```
Dense(..., activation='relu', ...)
```

<br>


## 파라미터 kernel_initializer, bias_initializer

https://www.tensorflow.org/api_docs/python/tf/keras/initializers

- 'he_normal'
- 'lecun_normal'

```
Dense(..., kernel_initializer='he_normal', bias_initializer='he_normal', ...)
```

<br>


## 라미터 kernel_regularizer, bias_regularizer

https://www.tensorflow.org/api_docs/python/tf/keras/regularizers

- l1()
- l1_l2()
- l2()

```
from keras.regularizers import l1
from keras.regularizers import l1_l2
from keras.regularizers import l2

Dense(..., kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1), ...)
```

<br>


# model.compile()

구조가 정의된 모델에 optimizer와 loss등을 설정하고 학습 실행할 준비를 한다.

https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile

```
model.compile(
    optimizer,
    loss,
    metrics,
    loss_weights,
)
```

<br>


## 파라미터 optimizer

웨이트들을 업데이트 하기 위한 값을 결정하는 방법

- 'SGD'
- 'RMSprop'
- 'Adam'
- 'Adagrad'
- ...

```
model.compile(..., optimizer='SGD', ...)
```

```
from tensorflow.keras import optimizers

sgd = optimizers.SGD(lr=0.001)
model.compile(..., optimizer=sgd, ...)
```

reference : https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

<br>


## 파라미터 loss

학습이 진행될 방향을 명시한 코스트 함수

- 'mean_squared_error' or 'mse'
- 'binary_crossentropy' : 클래스가 2개일 때
- 'categorical_crossentropy' : 클래스가 3개 이상일 때
- 'sparse_categorical_crossentropy' : 출력 값이 class index 일 때
- ...

```
model.compile(..., loss='mse', ...)
```


referene : https://www.tensorflow.org/api_docs/python/tf/keras/loss_weights

<br>


## 파라매터 metrics

학습 도중 성능을 보기 위한 지표.

- 'accuracy'
- ...

```
model.compile(..., metrics=['accuracy'], ...)
```


reference : https://www.tensorflow.org/api_docs/python/tf/keras/metrics

<br>


# 학습

model.fit()를 호출하여 실제 학습을 실행한다.

https://www.tensorflow.org/api_docs/python/tf/keras/Model

```
model.fit(
    x, # 입력
    y, # 출력
    epochs=100, 반복 횟수
    validation_data=..., # 학습 중에 성능을 보기 위한 데이터
    ...
)
```
<br>


# 모델 평가

테스틑를 위한 입력과 출력을 모델을 평가한다.

loss를 반환하며, metrics를 명시한 경우 해당 metric의 값을 반환한다.

```
loss = model.evaluate(test_x, test_y)

# metrics=['accuracy']를 주었을 경우
loss, acc = model.evaluate(test_x, test_y)
```

<br>


# 모델 사용

데이터에 대한 모델의 출력을 반환한다.

반환된 데이터의 shape는 (데이터 갯수, 출력노드 수)이다.

```
prediction = model.predict(test_x)
```


# 모델 저장과 로딩

https://www.tensorflow.org/beta/guide/keras/saving_and_serializing

## 모델 전체 저장과 로딩

```
model.save('my_model.h5')
new_model = keras.models.load_model('my_model.h5')
```

## 모델 설정만 저장과 로딩

```
config = model.get_config()
new_model = keras.Model.from_config(config)
```

```
json_config = model.to_json()
with open('model_config.json', 'w') as json_file:
    json_file.write(json_config)

with open('model_config.json') as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
```

## 모델 웨이트만 저장과 로딩

```
model.save_weights('model_weight.h5')
new_model.load_weights('model_weight.h5')
```

## TensorFlow 포멧의 저장과 로딩

```
keras.experimental.export_saved_model(model, 'model_path')

new_model = keras.experimental.load_from_saved_model('model_path')
```



# loss categorical_crossentropy 관련

2개의 사용 방법

- categorical_crossentropy
- sparse_categorical_crossentropy


## categorical_crossentropy

y의 값이 one hot encoding인 경우
```
1,0,0
0,1,0
0,0,1
```

출력 레이어 설정
```
model.add(Dense(3, activation="softmax")) # 출력 레이어
```

loss 설정
```
model.compile(..., loss='categorical_crossentropy')
```

## sparse_categorical_crossentropy

y의 값이 숫자 하나이며, category index 인 경우
```
0
1
2
```

출력 레이어 설정
```
model.add(Dense(3, activation="softmax")) # 출력 레이어. 1 이 아니라 3으로 설정
```

loss 설정
```
model.compile(..., loss='sparse_categorical_crossentropy')
```



