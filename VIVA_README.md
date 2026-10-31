# DL Practical – Viva Questions & Answers

Simple answers. Read the **Basics** section first, then each practical section.

---

## PART 0 – BASICS (asked in almost every viva)

**Q1. What is Deep Learning?**
A part of Machine Learning that uses neural networks with many layers ("deep") to learn patterns from data automatically, e.g. recognizing images, text, sound.

**Q2. Difference between ML and DL?**
- ML: we manually pick features; works well on small data.
- DL: the network learns features by itself; needs more data and computing power (GPU).

**Q3. What is a Neural Network?**
Layers of connected "neurons". Each neuron takes inputs, multiplies by weights, adds a bias, and passes the result through an activation function.
`output = activation(w1*x1 + w2*x2 + ... + b)`

**Q4. What are weights and bias?**
Numbers the network learns during training. Weights show how important each input is; bias shifts the output.

**Q5. What are the layers in a network?**
- Input layer – takes the data
- Hidden layers – learn patterns
- Output layer – gives the final answer

**Q6. What is an activation function? Why is it needed?**
A function applied to a neuron's output. It adds **non-linearity**, so the network can learn complex patterns. Without it, the whole network is just one straight-line equation.

**Q7. Name common activation functions.**
- **ReLU**: `max(0, x)` – used in hidden layers, fast and simple.
- **Sigmoid**: output between 0 and 1 – used for **binary** (yes/no) output.
- **Softmax**: gives probabilities for each class that add up to 1 – used for **multi-class** output.
- **Tanh**: output between -1 and 1.

**Q8. What is a loss function?**
It measures how wrong the prediction is. Training tries to make loss as small as possible.
- `sparse_categorical_crossentropy` – multi-class, labels are numbers (0,1,2…)
- `categorical_crossentropy` – multi-class, labels are one-hot
- `binary_crossentropy` – 2 classes
- `mse` (mean squared error) – regression / autoencoders

**Q9. Why sparse_categorical_crossentropy and not categorical_crossentropy?**
Our labels are plain integers (0–9). "Sparse" works directly with integers, so we don't need one-hot encoding.

**Q10. What is an optimizer?**
The algorithm that updates the weights to reduce loss.
- **SGD** (Stochastic Gradient Descent) – basic, updates weights using small batches.
- **Adam** – smarter; adjusts learning rate automatically; usually faster and better.

**Q11. What is Gradient Descent?**
A method to find minimum loss. It calculates the slope (gradient) of loss and moves the weights a small step in the downhill direction.

**Q12. What is learning rate?**
Size of each step during weight update. Too big → jumps around and doesn't learn. Too small → learns very slowly.

**Q13. What is Backpropagation?**
The way the network learns: error at the output is sent **backwards** through the layers to calculate how much each weight caused the error, then weights are updated.

**Q14. What is an epoch? batch size?**
- **Epoch**: one full pass over the whole training data.
- **Batch size**: number of samples processed before updating weights once (e.g. 32).

**Q15. What is overfitting? How to reduce it?**
Model does very well on training data but badly on new/test data (it memorised).
Fix: Dropout, more data, fewer epochs (early stopping), simpler model, data augmentation.

**Q16. What is underfitting?**
Model is too simple and does badly even on training data. Fix: bigger model, more epochs.

**Q17. What is Dropout?**
Randomly turns off some neurons (e.g. 50%) during training so the network doesn't depend on a few neurons → reduces overfitting.

**Q18. Why divide pixels by 255? (Normalization)**
Pixel values are 0–255. Dividing by 255 brings them to 0–1. Small, similar-range values make training faster and stable.

**Q19. What are training, validation and test data?**
- Training – used to learn.
- Validation – checked during training to see if the model is overfitting.
- Test – used only at the end to measure final performance.

**Q20. What is accuracy?**
Correct predictions ÷ total predictions.

**Q21. What is TensorFlow and Keras?**
- **TensorFlow** – Google's deep learning library (does the heavy maths).
- **Keras** – easy, high-level API on top of TensorFlow to build models in few lines.

**Q22. What is Sequential model?**
A Keras model where layers are stacked one after another in a straight line.

**Q23. What do compile(), fit(), evaluate(), predict() do?**
- `compile` – set optimizer, loss, metrics
- `fit` – train the model
- `evaluate` – test the model, returns loss and accuracy
- `predict` – give output for new data

**Q24. What is model.summary()?**
Prints all layers, output shapes and number of parameters (weights).

**Q25. What is a Dense layer?**
A fully connected layer – every neuron is connected to every neuron of the previous layer.

---

## PART 1 – FNN (Feedforward Neural Network) – 1A, 1B, 1C, 1E

**Q. What is a Feedforward Neural Network?**
Data flows only in one direction: input → hidden layers → output. No loops.

**Q. Explain your model.**
`Dense(128, relu) → Dense(64, relu) → Dense(10, softmax)`
Input is 784 pixels (28×28 image flattened). Output 10 neurons = 10 classes.

**Q. Why 784 inputs?** MNIST image is 28×28 = 784 pixels. FNN needs a flat 1-D input.

**Q. Why 3072 inputs in CIFAR10 (1C)?** 32 × 32 × 3 (RGB colour) = 3072.

**Q. Why 25 outputs in Sign Language MNIST (1E)?**
Labels go 0–24. Letter J (9) and Z (25) are missing because they need hand motion. So we keep 25 output units.

**Q. Why 10 outputs in last layer?** 10 classes (digits 0–9 / 10 clothing types / 10 CIFAR objects).

**Q. Why softmax in last layer?** To get the probability of each class; the highest one is the prediction.

**Q. What do the graphs show?**
Loss going down and accuracy going up over epochs = model is learning. If training accuracy keeps rising but validation accuracy falls → overfitting.

**Q. What is MNIST / Fashion MNIST / CIFAR10?**
- MNIST: 70,000 grayscale 28×28 images of handwritten digits 0–9.
- Fashion MNIST: same size, 10 clothing items (shirt, shoe, bag…).
- CIFAR10: 32×32 colour images, 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).

**Q. Why is FNN accuracy low on CIFAR10?**
FNN flattens the image and loses the position/shape information; colour images are complex. CNN works much better.

**Q. In 1C, why train_test_split?** Only one CSV was given, so we split it 80% train, 20% test.

---

## PART 2 – CNN (Convolutional Neural Network) – 2A, 2B, 2C

**Q. What is a CNN?**
A neural network designed for images. It uses filters to detect features like edges, shapes, and textures while keeping the spatial (position) information.

**Q. Why CNN better than FNN for images?**
- Keeps 2-D structure of image
- Fewer parameters (filters are shared)
- Detects features anywhere in the image

**Q. What is a Convolution layer (Conv2D)?**
A small filter (e.g. 3×3) slides over the image and multiplies values to detect a feature (edge, curve). `Conv2D(32, (3,3))` = 32 filters of size 3×3.

**Q. What is a filter / kernel?** A small matrix of weights that detects one feature.

**Q. What is Pooling (MaxPooling2D)?**
Reduces image size by taking the maximum value in each 2×2 block. Makes it faster and keeps the important features.

**Q. What is Flatten?** Converts the 2-D feature maps into a 1-D vector to give to Dense layers.

**Q. What is stride and padding?**
- Stride: how many pixels the filter moves each step.
- Padding "same": adds zeros at borders so output size = input size.

**Q. Explain your CNN.**
`Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(classes, softmax)`
Conv layers find features, pooling shrinks, Dense layers classify.

**Q. Why reshape to (28, 28, 1)?** CNN needs image shape: height, width, channels. Grayscale = 1 channel, colour = 3.

**Q. What are the 4 stages?**
1. Load & preprocess (normalize, reshape)
2. Define architecture
3. Train (fit)
4. Evaluate (accuracy, loss, graph)

**Q. What is the CIFAR10 reshape line in 2A?**
The CSV stores all red values, then green, then blue. `reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)` turns it into a proper 32×32 image with 3 colour channels at the end.

---

## PART 3 – AUTOENCODER (Anomaly Detection) – 3A, 3B, 3C, 3D, 3F

**Q. What is an Autoencoder?**
A neural network that learns to **copy its input to its output**. It has:
- **Encoder** – compresses the input into a small representation
- **Bottleneck** (latent space) – the compressed form
- **Decoder** – rebuilds the original input from it

**Q. Is it supervised or unsupervised?** Unsupervised – it doesn't need labels; input itself is the target. `model.fit(x, x)`

**Q. How does it detect anomalies?**
1. Train only on **normal** data.
2. It learns to rebuild normal data well (low error).
3. Abnormal data looks different, so it is rebuilt badly → **high reconstruction error**.
4. If error > threshold → anomaly.

**Q. What is reconstruction error?**
Difference between input and output. We use mean squared error: `mean((x - x_reconstructed)²)`

**Q. How did you choose the threshold?**
95th percentile of errors on normal training data – meaning only 5% of normal data is above it. Anything above is flagged.

**Q. Why loss = mse?** Output is numbers (not classes), so we measure squared difference between input and output.

**Q. 3A – Why train only on non-fraud data?**
Fraud is very rare (492 out of 2,84,807 ≈ 0.17%). This is **class imbalance**. A normal classifier would just say "not fraud" always. Autoencoder learns normal patterns, so fraud stands out.

**Q. 3A – Why StandardScaler?** Columns like Amount and Time have very large values compared to V1–V28. Scaling puts all features on the same scale (mean 0, std 1).

**Q. 3A – What are V1–V28?** Features already transformed by PCA to hide private customer details.

**Q. What is PCA?** Principal Component Analysis – reduces/transforms features into new uncorrelated ones; used here to anonymize data.

**Q. Why precision and recall instead of only accuracy (3A)?**
With 99.8% normal data, even a useless model gets 99.8% accuracy. Recall = how many frauds we caught; Precision = how many flagged were really fraud.

**Q. What is a confusion matrix?**
Table of: True Positive, True Negative, False Positive, False Negative.

**Q. 3B – What is a convolutional autoencoder?**
Uses Conv + MaxPooling to compress the image (encoder) and Conv + UpSampling to rebuild it (decoder). Good for images.

**Q. What is UpSampling?** Opposite of pooling – makes the image bigger (repeats pixels).

**Q. 3B – MVTec AD?** Images of industrial objects (bottle, screw, cable…) — normal and defective. We train on normal (good) images; defects give high error.

**Q. 3C, 3D – What is LSTM?**
Long Short-Term Memory – a type of RNN (Recurrent Neural Network) that remembers information over time. Used for **sequence/time-series** data.

**Q. What is RNN?** A network with a loop that passes information from one time step to the next. Simple RNN forgets long-term info (vanishing gradient); LSTM fixes this using gates.

**Q. What are LSTM gates?** Forget gate (what to throw away), Input gate (what to store), Output gate (what to output).

**Q. Explain LSTM autoencoder.**
`LSTM(32) → RepeatVector(10) → LSTM(32, return_sequences=True) → TimeDistributed(Dense)`
- First LSTM compresses the whole sequence into one vector (encoder).
- RepeatVector copies it for each time step.
- Second LSTM + TimeDistributed rebuild the sequence (decoder).

**Q. What is a time window?**
Group of consecutive readings (we use 10). LSTM learns patterns over these 10 steps.

**Q. Why MinMaxScaler?** Scales each feature to 0–1 so no feature dominates.

**Q. 3C – Why ffill()?** Forward fill – fills a missing value with the previous value (good for time-series).

**Q. 3D – NASA Turbofan?** Sensor readings (temperature, pressure, speed…) of jet engines over cycles until failure. Fault_Label = 1 means faulty.

**Q. 3D – Why drop unit, cycle, RUL?** They are IDs/counters, not sensor readings. RUL (Remaining Useful Life) would directly leak the answer.

**Q. 3F – How is Y detected?**
Trained on all letters except Y (label 24). Y was never seen, so it gets higher reconstruction error → flagged as anomaly.

---

## PART 4 – CBOW / Word2Vec – 4A, 4B, 4C, 4D

**Q. What is NLP?** Natural Language Processing – making computers understand human text/language.

**Q. What is a word embedding?**
A list of numbers (vector) representing a word. Words with similar meaning have similar vectors. e.g. "king" and "queen" are close.

**Q. Why not one-hot encoding?** One-hot vectors are huge (size = vocabulary) and don't show any meaning/similarity between words. Embeddings are small and meaningful.

**Q. What is Word2Vec?**
A model (by Google) that learns word embeddings from text. Two types:
- **CBOW** (Continuous Bag of Words) – predicts the **target word from surrounding context words**.
- **Skip-gram** – predicts **context words from the target word**.

**Q. CBOW example?** Sentence "the cat sat on mat": context [the, cat, on, mat] → predict "sat".

**Q. CBOW vs Skip-gram?** CBOW is faster and good for frequent words. Skip-gram is better for rare words and small data.

**Q. What does sg=0 mean?** sg=0 → CBOW, sg=1 → Skip-gram.

**Q. Explain Word2Vec parameters.**
- `vector_size=100` – each word becomes 100 numbers
- `window=5` – 5 words on each side are taken as context
- `min_count=2` – ignore words that appear less than 2 times
- `epochs` – number of training passes

**Q. What preprocessing did you do?**
Lowercase → remove HTML/links/symbols → split into words (tokenize) → remove stop words.

**Q. What are stop words?** Very common words with little meaning: "the, is, a, an, and". Removed to keep important words.

**Q. What is tokenization?** Splitting text into words (tokens).

**Q. What is vocabulary?** The set of all unique words the model knows.

**Q. 4A – How did you classify documents?**
1. Train Word2Vec CBOW on the reviews.
2. Each review = **average** of embeddings of all its words → one 100-number vector.
3. Feed this to a Dense neural network → sigmoid output (positive/negative).

**Q. 4A – Why sigmoid + binary_crossentropy?** Only 2 classes (positive / negative).

**Q. 4A – Why no padding?** Padding makes all sequences the same length. Since we average embeddings, every review already becomes the same size (100), so padding isn't needed.

**Q. What does most_similar() do?** Returns words whose vectors are closest (by cosine similarity) to the given word.

**Q. What is cosine similarity?**
Measures the angle between two vectors. 1 = same direction (very similar), 0 = unrelated, -1 = opposite.

**Q. 4D – How does recommendation work?**
1. Each tweet → average word vector.
2. User's input keywords → average word vector.
3. Cosine similarity between input and every tweet.
4. Show the top 5 most similar tweets.

**Q. What is content-based recommendation?** Recommending items similar in **content** to what the user likes/searched (vs collaborative filtering which uses other users' behaviour).

---

## PART 5 – TRANSFER LEARNING (VGG16) – 5A, 5B, 5C

**Q. What is Transfer Learning?**
Using a model already trained on a big dataset (ImageNet, 14 million images, 1000 classes) and reusing its knowledge for our smaller dataset. Saves time and gives better accuracy with less data.

**Q. What is VGG16?**
A CNN with 16 layers (13 conv + 3 dense) made by Oxford's Visual Geometry Group. Uses only 3×3 filters. Pre-trained on ImageNet.

**Q. What is ResNet?** Residual Network – very deep CNN that uses "skip connections" so deep networks can train without vanishing gradient.

**Q. What is include_top=False?** Removes VGG16's original final Dense layers (made for 1000 ImageNet classes), so we can add our own for our classes.

**Q. What is the .h5 file?** The saved pre-trained VGG16 weights (without top). Loaded from file so it works offline.

**Q. Why freeze base layers (base.trainable = False)?**
The base already knows how to detect edges, shapes, textures. Freezing keeps that knowledge and we only train our new layers. Faster and avoids destroying learned features.

**Q. What custom layers did you add?**
`Flatten → Dense(256, relu) → Dropout(0.5) → Dense(num_classes, softmax)`

**Q. What is Fine-tuning?**
After training the new layers, we **unfreeze the top few layers** (last 4) of VGG16 and train again with a **very small learning rate** (1e-5) so they adapt to our dataset.

**Q. Why small learning rate during fine-tuning?** A big one would destroy the useful pre-trained weights.

**Q. Feature extraction vs fine-tuning?**
- Feature extraction: base frozen, only new layers trained.
- Fine-tuning: some base layers also trained.

**Q. What is preprocess_input?** Prepares images the same way VGG16 was trained (converts RGB→BGR and subtracts ImageNet mean). Needed for correct results.

**Q. What is image_dataset_from_directory?** Keras function that loads images from folders; each folder name becomes the class label.

**Q. 5C – Why resize Fashion MNIST to 64×64 and RGB?** VGG16 needs at least 32×32 and 3 colour channels; Fashion MNIST is 28×28 grayscale.

**Q. Datasets?**
- Caltech-101: 101 object categories (+ background), ~9,000 images.
- Intel Image Classification: 6 scenes – buildings, forest, glacier, mountain, sea, street.
- Fashion MNIST: 10 clothing types.

**Q. Classification vs Object Detection?**
- Classification: says **what** is in the image.
- Object detection: says what AND **where** (draws bounding boxes), e.g. YOLO, SSD, Faster R-CNN.
- (Our practical is actually image classification using transfer learning.)

---

## QUICK LAST-MINUTE CHEAT SHEET

| Thing | Use |
|---|---|
| ReLU | hidden layers |
| Softmax | last layer, many classes |
| Sigmoid | last layer, 2 classes |
| sparse_categorical_crossentropy | many classes, integer labels |
| binary_crossentropy | 2 classes |
| mse | autoencoder / regression |
| Adam / SGD | optimizers |
| Dropout | reduce overfitting |
| Conv2D | find features in images |
| MaxPooling | shrink image |
| Flatten | 2D → 1D |
| LSTM | sequences / time series |
| Autoencoder | input = output, anomaly detection |
| sg=0 | CBOW |
| include_top=False | remove VGG's old classifier |
| trainable=False | freeze layers |
