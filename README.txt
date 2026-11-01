DL PRACTICAL EXAM - README
==========================

SETUP (do this once)
  pip install tensorflow pandas scikit-learn matplotlib gensim

HOW TO RUN
  Always go INSIDE the folder first, then run (the code reads the data from the same folder):
    cd 1/1a
    python3 code.py

NOTE: All code was syntax-checked but NOT run (TensorFlow was not installed).
      Run each one at least once before the exam.


!!! PROBLEMS - MUST TAKE CARE !!!
---------------------------------
3B  (MVTec AD anomaly detection)
    -> DATASET MISSING. Teacher's folder does NOT have MVTec AD.
    -> Download one category (e.g. "bottle") from Kaggle and put it in 3/3b like:
         3/3b/bottle/train/good/*.png
         3/3b/bottle/test/good/*.png
         3/3b/bottle/test/<defect name>/*.png
    -> If you use another category, change  folder = "bottle"  in code.py.
    -> Ask the teacher if they will give this dataset in the lab.

1B  (FNN Fashion MNIST)
    -> Statement step (b) says "CIFAR10" - this is a TYPO.
    -> Title says Fashion MNIST, so code uses Fashion MNIST.

1D
    -> No 1D statement was given, so there is no 1D folder.


THINGS TO REMEMBER / TEACHER MAY ASK
------------------------------------
1C, 2A  (CIFAR10)
    -> Only ONE csv given (no separate test file). Code splits it 80% train / 20% test
       using train_test_split.
    -> 2A (CNN): csv stores all Red pixels, then Green, then Blue, so reshape is:
         x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

1E, 2C, 3F  (Sign Language MNIST)
    -> Labels are 0-24 but label 9 (J) is missing (J and Z need motion).
    -> So output layer has 25 units, real classes = 24.
    -> 3F: Y = label 24 is the anomaly; trained on the other 23 classes.

3C, 3D  (LSTM autoencoder)
    -> LSTM needs sequences, so data is cut into windows of 10 time steps.
    -> 3C energy data is already 0-1 and has no missing values, but code still does
       ffill() + MinMaxScaler because the statement asks for it.

4A  (CBOW IMDB)
    -> Uses teacher's "IMDB Dataset.csv", NOT  keras.datasets.imdb
       (that one needs internet and is already numbers, so you can't remove stop words).
    -> No padding: averaging word embeddings gives every review the same size vector,
       so padding is not needed. Say this if asked.

4B, 4D  (Twitter)
    -> The twitter dataset is fake auto-generated text, so similar-word results may
       look random. 4B picks example words automatically so it never crashes.
    -> 4D asks you to TYPE keywords when it runs (e.g.  sell economy development).

4A-4D
    -> sg=0 means CBOW (sg=1 would be Skip-gram). Teacher will look for this.
    -> Needs gensim installed.

5A, 5B, 5C  (Transfer learning VGG16)
    -> Uses the teacher's vgg16_weights...notop.h5 file, so works OFFLINE.
    -> All 3 codes are the same; only the DATA line changes.
    -> 5C: teacher's folder had an extra "train/Ankle-boot" subfolder which would become
       an 11th wrong class. It was NOT copied. All images are resized to 64x64 (VGG needs >= 32).
    -> 5A: Caltech-101 has 102 folders (includes BACKGROUND_Google). Slowest to train -
       reduce epochs if it takes too long.


FOLDER SUMMARY
--------------
1  FNN    : 1a MNIST | 1b Fashion MNIST | 1c CIFAR10 | 1e Sign MNIST
2  CNN    : 2a CIFAR10 | 2b MNIST | 2c Sign MNIST
3  AutoEnc: 3a CreditCard | 3b MVTec (missing) | 3c Energy | 3d NASA Turbofan | 3f Sign MNIST (Y anomaly)
4  CBOW   : 4a IMDB | 4b Twitter | 4c document3 | 4d Twitter recommendation
5  VGG16  : 5a Caltech-101 | 5b Intel Images | 5c Fashion MNIST images
