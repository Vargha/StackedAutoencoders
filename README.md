# StackedAutoencoders
Stacked Autoencoders in Image Classification
For more in depth explanation visit: https://www.linkedin.com/pulse/stacked-autoencoders-image-classification-vargha-hokmran

The CIFAR-10 dataset has 50,000 train images and 10,000 test images. This can take long.
Prepare a smaller training set: 10,000 training samples and 5,000 test samples.

Convert the images to grayscale and train fully connected autoencoders.
Train a stacked autoencoder with 3 such layers:
* First autoencoder: [1024 → 1000 → 1024].
* Second autoencoder: [1000 → 800 → 1000].
* Third autoencoder: [800 → 500 → 800].
The stacked encoder is then represented as [1024,1000,800,500]
* Add a classiﬁer on top of the ﬁnal layer and ﬁne-tune the stacked autoencoder with labeled samples.

Add a classiﬁer on top of the ﬁnal layer and ﬁne-tune the stacked autoencoder with labeled samples.
For the ﬁne-tuning consider:
(i) 10-labeled sample per class
(ii) 100-labeled samples per class.
Evaluate the classiﬁcation accuracies on the test dataset using these models.
You need to lower the learning rates of the stacked layers or freeze them and only train the weights for the classiﬁcation layer.
