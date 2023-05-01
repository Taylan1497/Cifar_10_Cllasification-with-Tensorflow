# Cifar_10_Cllasification-with-Tensorflow

Main.py

-Run main.py in the terminal (python main.py) .
-This is the training part with a batch size of 32, epoch number of 30 and learning
rate 0.01.
-It will also generate loss and accuracy curves and save model.
-If you want to see tsne plotting or early stopping, please edit Main.py such that
early_stopping=True and tsne_plot=True.
-In the same directory data should be inside sample_data/cifar_10_data/
it is loaded from the link https://www.cs.toronto.edu/~kriz/cifar.html
-In the same directory there should be Plots folder to save loss and accuracy
curves, sample_data/model_save folder to keep model save files.

Eval.py

- Run eval.py in the terminal (python eval.py)

- It will generate test data as inputs, load the saved model and then generate test
accuracy results.

Model.py

-The CNN architecture is in this python file. It is called in main.py and eval.py.

Eval_BackUp.ipynb

- If you see any package dependency problem you can run Eval_BackUp.ipynb on jupyter notebook. To see test results.


