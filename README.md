# 교육 목표

딥러닝을 이해하고 실제 문제에 적용하여 해결할 수 있다.


<br>

# 교육 상세 목표

- 딥러닝을 이해한다.
- 딥러닝을 적용할 수 있는 문제를 이해한다.
- 딥러닝을 실 문제에 적용하는 방법을 이해한다.
- 딥러닝에 적용하기 위한 데이터 처리방법을 이해한다.
- 데이터 처리방법을 파악하고 구현할 수 있다.
- 딥러닝을 적용하여 실제 문제를 해결한다.


<br>

# 대상

선수 교육(python, numpy, pandas)를 아는 개발자.




# 일자별 계획

## 1일차

딥러닝 이해

- 딥러닝 개념 : [deep_learning_intro.pptx](material/deep_learning/deep_learning_intro.pptx)



<br>

## 2일차

Keras로 구현한 딥러닝 코드

- DNN in Keras : [dnn_in_keras.ipynb](material/deep_learning/dnn_in_keras.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/dnn_in_keras.ipynb)
    - 표준 Keras 딥러닝 코드
    - 로스 보기
    - 은닉층과 노드 수
    - batch size와 학습
    - 데이터 수와 학습
    - normalization
    - 모델 저장과 로딩
    - 웨이트 초기값
    - 노이즈 내구성
    - GPU 설정
    - 데이터 수와 overfitting : [data_count_and_overfitting.ipynb](material/deep_learning/data_count_and_overfitting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/data_count_and_overfitting.ipynb)
    - overfitting 처리하기 : [treating_overfitting.ipynb](material/deep_learning/treating_overfitting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/treating_overfitting.ipynb)
    - callback : [callback.ipynb](material/deep_learning/callback.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/callback.ipynb)
    - 다양한 입출력

<br>

## 3일차

분류기로서의 DNN 

- 분류기로서 DNN : [dnn_as_a_classifier.ipynb](material/deep_learning/dnn_as_a_classifier.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/dnn_as_a_classifier.ipynb)
- IRIS 분류: [dnn_iris_classification.ipynb](material/deep_learning/dnn_iris_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/dnn_iris_classification.ipynb)
- MNIST 분류 : [dnn_mnist.ipynb](material/deep_learning/dnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/dnn_mnist.ipynb)


<br>

CNN
- MNIST 영상분류 : [cnn_mnist.ipynb](material/deep_learning/cnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/cnn_mnist.ipynb)
- CIFAR10 컬러영상분류 : [cnn_cifar10.ipynb](material/deep_learning/cnn_cifar10.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/cnn_cifar10.ipynb)
- IRIS 분류 : [iris_cnn.ipynb](material/deep_learning/iris_cnn.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/iris_cnn.ipynb)


<br>

## 4일차

다양한 활용
- 디노이징 AutoEncoder : [denoising_autoencoder.ipynb](material/deep_learning/denoising_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/denoising_autoencoder.ipynb)
- Super Resolution : [mnist_super_resolution.ipynb](material/deep_learning/mnist_super_resolution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/mnist_super_resolution.ipynb)


<br>

실제 영상인식
- Data Augmentation : [data_augmentation.ipynb](material/deep_learning/data_augmentation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/data_augmentation.ipynb)
- VGG로 영상 분류, 전이학습 : [VGG16_classification_and_cumtom_data_training.ipynb](material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb)


<br>

## 5일차

Functional API

- functional api : [functional_api.ipynb](material/deep_learning/functional_api.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/functional_api.ipynb)


다양한 활용
- 영역 분할(segmentation) - U-Net : [lung_segementation.ipynb](material/deep_learning/lung_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/lung_segementation.ipynb)
- AutoEncoder를 사용한 비정상 탐지 : [anomaly_detection_using_autoencoder.ipynb](material/deep_learning/anomaly_detection_using_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/anomaly_detection_using_autoencoder.ipynb)


<br>

- 영상 분할(Segementation)
  - U-Net을 사용한 : [unet_segementation.ipynb](material/deep_learning/unet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/unet_segementation.ipynb)
  - M-Net을 사용한 : [mnet_segementation.ipynb](material/deep_learning/mnet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/mnet_segementation.ipynb)
<br>

## 6일차


다양한 활용
- 얼굴 인식(face recognition) : [Face_Recognition.ipynb](material/deep_learning/Face_Recognition.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/Face_Recognition.ipynb)


<br>

RNN, 강화학습
- RNN 이해하기 : [deep_learning_intro.pptx](material/deep_learning//deep_learning_intro.pptx)
- CNN, RNN 사용한 text 분류 : [text_classification.ipynb](material/deep_learning/text_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/text_classification.ipynb)
- RNN을 사용한 MNIST 분류 : [mnist_rnn.ipynb](material/mnist_rnn.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/mnist_rnn.ipynb)
- 강화학습 이해하기 : [deep_learning_intro.pptx](material/deep_learning//deep_learning_intro.pptx)


<br>

- 알파고 이해하기 : [understanding_ahphago.pptx](material/deep_learning/understanding_ahphago.pptx)

<br>

## 7일차

GAN

- GAN 이해하기 : [deep_learning_intro.pptx](material/deep_learning//deep_learning_intro.pptx), 
- GAN MNIST 학습 : [wgan_pg_mnist.ipynb](material/deep_learning/wgan_pg_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/wgan_pg_mnist.ipynb)
- Cycle GAN 리뷰 : [cycle_gan.pdf](material/deep_learning/cycle_gan.pdf)


<br>



## 8일차


Object Detection

- YOLO : [object_detection.md](material/deep_learning/object_detection.md)

<br>



## 9일차


다양한 활용
- 얼굴 감정 분류 : [face_emotion_classification.ipynb](material/deep_learning/face_emotion_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/face_emotion_classification.ipynb)
- 얼굴 탐지 : [track_faces_on_video_realtime.ipynb](material/deep_learning/track_faces_on_video_realtime.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/track_faces_on_video_realtime.ipynb)
- 화재 영상 분류 : [Fire.tar.gz](material/deep_learning/Fire.tar.gz), [spatial_envelope_static_8outdoorcategories.tar.gz](material/deep_learning/spatial_envelope_static_8outdoorcategories.tar.gz)
- 포즈 추출 : [pose_extraction_using_open_pose.ipynb](material/deep_learning/pose_extraction_using_open_pose.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/pose_extraction_using_open_pose.ipynb)


<br>

Kaggle 문제 풀이 설명


- Intel Image Classification
    - Basic CNN : https://www.kaggle.com/uzairrj/beg-tut-intel-image-classification-93-76-accur
    - DenseNet with focalloss : https://www.kaggle.com/youngtaek/intel-image-classification-densenet-with-focalloss
- Google Landmark Retrieval 2020
    - Create and Train ResNet50 from scratch : https://www.kaggle.com/sandy1112/create-and-train-resnet50-from-scratch
- Car Image segmentation
	- keras U-Net : https://www.kaggle.com/ecobill/u-nets-with-keras
	- VGG16 U-Net : https://www.kaggle.com/kmader/vgg16-u-net-on-carvana

<br>


## 10일차

딥러닝 교육 

- SW 산업에 대하여
- SW 개발 이란
- 딥러닝 개발 계획 리뷰
- 채용 공고 리뷰

<br>


## 기타

교육환경, numpy, pandas, matplot
- 교육 환경 : [env.md](material/env.md)
- numpy : 데이터 로딩, 보기, 데이터 변환, 형태 변경 : [library.md](material/library.md)
- linux 기본 명령어 : 
    - bash, cd, ls, rm, mkdir, mv, tar, unzip
    - docker, pip, apt, wget, EVN, git
    - 교육 자료
        - [linux.md](material/linux.md)
        - [linux_exercise.md](material/linux_exercise.md)
- pandas, matplot : [ibrary.md](material/library.md)

기타 자료
- [의학논문 리뷰](https://docs.google.com/presentation/d/1SZ-m4XVepS94jzXDL8VFMN2dh9s6jaN5fVsNhQ1qwEU/edit)
- GCP에 VM 생성하고 Colab 연결하기 : [create_GCP_VM.pdf](material/deep_learning/create_GCP_VM.pdf)
- 흥미로운 딥러닝 결과 : [some_interesting_deep_learning.pptx](material/deep_learning/some_interesting_deep_learning.pptx)
- yolo를 사용한 실시간 불량품 탐지 : https://drive.google.com/file/d/194UpsjG7MyEvWlmJeqfcocD-h-zy_4mR/view?usp=sharing
- YOLO를 사용한 자동차 번호판 탐지 : https://drive.google.com/file/d/1jlKzCaKj5rGRXIhwMXtYtVnx_XLauFiL/view?usp=sharing
- GAN을 사용한 생산설비 이상 탐지 : [anomaly_detection_using_gan.pptx](material/deep_learning/anomaly_detection_using_gan.pptx)
- 이상탐지 동영상 : [drillai_anomaly_detect.mp4](material/deep_learning/drillai_anomaly_detect.mp4)
- 훌륭한 논문 리스트 : https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap
- online CNN 시각화 자료 : https://poloclub.github.io/cnn-explainer/
- GradCAM : [grad_cam.ipynb](material/deep_learning/grad_cam.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hongik_2020/blob/master/material/deep_learning/grad_cam.ipynb)

<br>

# 딥러닝 활용을 위한 지식 구조

```
Environment
    jupyter
	colab
	usage
		!, %, run
    GCP virtual machine
linux
	ENV
	command
		cd, pwd, ls
		mkdir, rm, cp
		head, more, tail, cat
	util
		apt
		git, wget
		grep, wc, tree
		tar, unrar, unzip
	gpu
		nvidia-smi

python
	env
		python
			interactive
			execute file
		pip
	syntax
        variable
        data
            tuple
            list
            dict
            set
        loop
        if
        comprehensive list
        function
        class
	module
		import

libray
    numpy
        load
        op
        shape
        slicing
        reshape
        axis + sum, mean
    pandas
        load
        view
        to numpy
    matplot
        draw line graph
        scatter
        show image

Deep Learning
    DNN
        concept
            layer, node, weight, bias, activation
            cost function
            GD, BP
        data
            x, y
            train, validate, test
            shuffle
        learning curve : accuracy, loss
        tunning
            overfitting, underfitting
            regularization, dropout, batch normalization
            data augmentation
        Transfer Learning
    type
        supervised
        unsupervised
        reinforcement
    model
        CNN
            varnilla, VGG16
        RNN
        GAN
    task
        Classification
        Object Detection
        Generation
	Segmentation
	Pose Extraction
	Noise Removing
	Super Resolution
	Question answering
	Auto Captioning
        target : text/image

TensorFlow/Keras
    basic frame
        data preparing
            x, y
            train, valid, test
            normalization
            ImageDataGenerator
        fit
        evaluate
        predict
    model
        activation function
        initializer
    tuning
        learning rate
        regularier
        dropout
        batch normalization
    save/load
    compile
        optimizer
        loss
        metric
```

 
