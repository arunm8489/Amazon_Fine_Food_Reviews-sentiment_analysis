
<img src="https://github.com/arunm8489/Amazon_Fine_Food_Reviews-sentiment_analysis/blob/master/img/amazonfood.png" alt="amazon" width="1000" height="300">

# Amazon_Fine_Food_Reviews-sentiment_analysis


Amazon Fine Food Reviews is sentiment analysis problem where we classify each review as positive and negative using machine learning and deeplearning techniques. Finally we will deploy our best model using Flask.

**Data Source**: https://www.kaggle.com/snap/amazon-fine-food-reviews

**Objective**: Given a review, determine whether the review is positive (Rating of 4 or 5) or negative (rating of 1 or 2).

**Blog explanation**: 
I have written a blog explaining the approach i used to solve this sentiment analysis problem from basic EDA to deployment. You can read my medium blog for that.

**Results**:

<img src="https://github.com/arunm8489/Amazon_Fine_Food_Reviews-sentiment_analysis/blob/master/img/result_1.png" alt="amazon" width="700">

<img src="https://github.com/arunm8489/Amazon_Fine_Food_Reviews-sentiment_analysis/blob/master/img/result_2.png" alt="amazon" width="700">

We got a better generalized model using LSTM with 2 layers.

**Usage**


1. First install the requirements

> pip install -r requirements.txt

2. Download the weights from <a href='https://drive.google.com/drive/folders/13QNhlb-HA_3Mn42yGdQSYBDVCdZ7316S?usp=sharing'>drive</a> and move the contents to 'weights' folder.

3. Start our applicaion by running

> python app.py 

note:
After cloning the repository if you donot need the notebooks, you are free to delete Notebook folder.
You can also test our application using tmp.py

> python tmp.py
