# https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/ner/ner_model.py

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from simpletransformers.ner import NERModel,NERArgs

data = pd.read_csv("ner_dataset.csv",encoding="latin1" )
data = data.fillna(method ="ffill")


data["Sentence #"] = LabelEncoder().fit_transform(data["Sentence #"] )
data.rename(columns={"Sentence #":"sentence_id","Word":"words","Tag":"labels"}, inplace =True)
data["labels"] = data["labels"].str.upper()
X = data[["sentence_id","words"]]
Y = data["labels"]

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size =0.2)
train_data = pd.DataFrame({"sentence_id":x_train["sentence_id"],"words":x_train["words"],"labels":y_train})
test_data = pd.DataFrame({"sentence_id":x_test["sentence_id"],"words":x_test["words"],"labels":y_test})

label = data["labels"].unique().tolist()


args = NERArgs()
args.num_train_epochs = 1
args.learning_rate = 1e-4
args.overwrite_output_dir =True
args.train_batch_size = 32
args.eval_batch_size = 32

model = NERModel('bert', 'bert-base-cased',labels=label,args=args)
model.train_model(train_data, eval_data=test_data, acc=accuracy_score)
result, model_outputs, preds_list = model.eval_model(test_data)

print(result)

prediction, model_output = model.predict(["What is the new name of Bangalore"])

print(prediction)