import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pickle as pkl
class DataEncoding:
  def __init__(self,data):
    self.data=data 
  def label_encoder(self,columns):
    label_encoder=LabelEncoder()
    for i in columns:
      self.data[i]=label_encoder.fit_transform(self.data[i])
    return self.data
  def one_hot_encoder(self,columns):
    one_hot_encoder=OneHotEncoder(sparse_output=False)
    encoded_data=one_hot_encoder.fit_transform(self.data[columns])
    encoded_columns=one_hot_encoder.get_feature_names_out(columns)
    encoded_df=pd.DataFrame(encoded_data,columns=encoded_columns)
    self.data=pd.concat([self.data.drop(columns,axis=1),encoded_df],axis=1)
    return self.data
  def save_label_encoding(self,path):
    with open(path,'wb') as f:
      pkl.dump(self.label_encoder,f)
  def save_one_hot_encoding(self,path):
    with open(path,'wb') as f:
      pkl.dump(self.one_hot_encoder,f)
