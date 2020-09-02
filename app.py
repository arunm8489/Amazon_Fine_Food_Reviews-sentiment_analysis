from flask import Flask, request, jsonify
import pickle,os,json
# replacing some phrases like won't with will not
import numpy as np
import re,torch
import bs4
import torch.nn as nn
import torch.nn.functional as F
from model import Network


app = Flask(__name__)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def model_predict(sentance,seq_len,corpus_dict_,model):
  sentance = re.sub(r"http\S+", "", sentance)
  sentance = bs4.BeautifulSoup(sentance, 'lxml').get_text()
  sentance = decontracted(sentance)
  # removing extra spaces and numbers
  sentance = re.sub("\S*\d\S*", "", sentance).strip()
  # removing non alphabels
  sentance = re.sub('[^A-Za-z]+', ' ', sentance)
  text = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)

  seq = [corpus_dict_[word] for word in text.split() if word in corpus_dict_.keys()]

  #padding
  padd = np.zeros((1,seq_len),dtype=int)
  padd[0][-len(seq):] = seq
  padd = torch.from_numpy(padd)
  #prediction
  padd = padd.to(device)
  padd = padd.long()
  y_pred = model(padd)

  #returning probability and class
  prob = y_pred[:,-1]
  prob = F.sigmoid(prob)
  prob = prob.detach().cpu().numpy()[0]
  label = 'positive' if prob >= 0.5 else 'negative'
  prob = prob if label == 'positive' else 1-prob
  return [label,prob]

#paths
stopwords_path = os.path.join('weights','stopwords.pkl')
weight_path = os.path.join('weights','weight')
embedding_path = os.path.join('weights','embedding_matrix_model.npy')
corpus_path = os.path.join('weights','corpus_dict')

#loading corpus,stopwords
stopwords = pickle.load(open(stopwords_path,"rb"))
embedding_matrix = np.load(embedding_path)

with open(corpus_path) as json_file: 
    corpus_dict = json.load(json_file)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Network(embedding_matrix=embedding_matrix,hidden_dim=100,no_layers=2)

#maping the location since we are training on gpu environment
checkpoint = torch.load(weight_path,map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()



@app.route('/',methods=['POST'])
def predict():
    text = request.get_json(force=True)
    name = text['review']
    class_name,prob = model_predict(sentance=name,seq_len=225,corpus_dict_=corpus_dict,model=model)
    response = {'class_name':class_name,'class_probability':str(prob)}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)