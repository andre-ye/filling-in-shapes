# filling-in-shapes
This repository contains code for a shape-filling AI. A fun exercise for learning about semantic segmentation.
- `generating-shapes.py`: code for creating images of shapes in the dataset.
- `training-model.py`: code for training a UNet-style neural network on the data.
- `prediction.py`: code for displaying predictions on a folder of images.
- `model-weights.h5`: file hosting the weights of the trained UNet model.

10k and 25k sample versions of the dataset were uploaded to [Kaggle](https://www.kaggle.com/washingtongold/filling-in-shapes?select=10k-sample-dataset-w-identical-io). The dataset includes:
- *Shapes*: squares, rectangles, ellipses, circles, stars, filled circles/dots, lines/not closed shapes
- *Picture*: 128x128 grayscale PNG; padding of 10 on sides; pictures have an outline thickness of 4
- *Transformations*: random shifts and rotations

Examples of model performance on validation data:
![image](https://user-images.githubusercontent.com/73039742/113469564-5366f080-9403-11eb-91e4-4901e3a50a8e.png)
![image](https://user-images.githubusercontent.com/73039742/113469560-4b0eb580-9403-11eb-9903-bcaa14be928c.png)
