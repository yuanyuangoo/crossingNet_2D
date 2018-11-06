from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sys
sys.path.append('./data/')
import ref
Keys = np.array(list(ref.tags.keys()))
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Keys)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])

print(label_encoder.transform('Directions'))