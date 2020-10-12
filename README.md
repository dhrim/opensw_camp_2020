# 교육 목표

딥러닝 개념을 파악하고 실제 동작하는 딥러닝을 경험한다.


<br>

# 교육 상세 목표

- 딥러닝의 개념을 이해한다.
- 딥러닝으로 해결할 수 있는 문제를 이해한다.
- 딥러닝에 적용하기 위한 데이터를 이해한다.
- Keras로 구현한 딥러닝 코드를 파악하고 실행해 본다.
- 딥러닝을 적용한 작업들을 파악한다.


<br>

# 대상

프로그래밍 경험이 없는 대학생


<br>

# 프로그램

- 딥러닝 개념 : [deep_learning_intro.pptx](material/deep_learning/deep_learning_intro.pptx)

<br>

- Keras로 구현한 딥러닝 코드[dnn_in_keras_shortly.ipynb](material/deep_learning/dnn_in_keras_shortly.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/opensw_camp_2020/blob/master/material/deep_learning/dnn_in_keras_shortly.ipynb)


<br>

- IRIS 분류: [dnn_iris_classification.ipynb](material/deep_learning/dnn_iris_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/opensw_camp_2020/blob/master/material/deep_learning/dnn_iris_classification.ipynb)
- MNIST 분류 : [dnn_mnist.ipynb](material/deep_learning/dnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/opensw_camp_2020/blob/master/material/deep_learning/dnn_mnist.ipynb)

<br>

- CNN MNIST 영상분류 : [cnn_mnist.ipynb](material/deep_learning/cnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/opensw_camp_2020/blob/master/material/deep_learning/cnn_mnist.ipynb)
- CNN CIFAR10 컬러 영상분류 : [cnn_cifar10.ipynb](material/deep_learning/cnn_cifar10.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/opensw_camp_2020/blob/master/material/deep_learning/cnn_cifar10.ipynb)

<br>

- AutoEncoder :  [autoencoder.ipynb](material/deep_learning/autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/opensw_camp_2020/blob/master/material/deep_learning/autoencoder.ipynb)
- 디노이징 AutoEncoder : [denoising_autoencoder.ipynb](material/deep_learning/denoising_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/opensw_camp_2020/blob/master/material/deep_learning/denoising_autoencoder.ipynb)
- Super Resolution : [mnist_super_resolution.ipynb](material/deep_learning/mnist_super_resolution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/opensw_camp_2020/blob/master/material/deep_learning/mnist_super_resolution.ipynb)


<br>

- U-Net을 사용한 영상 분할: [unet_segementation.ipynb](material/deep_learning/unet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/opensw_camp_2020/blob/master/material/deep_learning/unet_segementation.ipynb)

<br>

- RNN, GAN, 강화학습 이해하기 : [deep_learning_intro.pptx](material/deep_learning//deep_learning_intro.pptx)
- 알파고 이해하기 : [understanding_ahphago.pptx](material/deep_learning/understanding_ahphago.pptx)

<br>

- 흥미로운 딥러닝 결과 : [some_interesting_deep_learning.pptx](material/deep_learning/some_interesting_deep_learning.pptx)
- online CNN 시각화 자료 : https://poloclub.github.io/cnn-explainer/
- online 딥러닝 플랫폼 : https://app.deepcognition.ai/


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

 
