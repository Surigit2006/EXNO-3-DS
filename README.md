## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
!pip install --upgrade category_encoders
```
```
import pandas as pd
df=pd.read_csv(r"C:\Users\Suriya\Downloads\Encoding Data.csv")
df
```

![image](https://github.com/user-attachments/assets/020339e9-8fd6-4b2b-bab3-ebf45c54cb9f)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/63c875d6-2e8d-480c-bfa8-8c48e897ce6b)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/2c07e0db-9e63-4272-88a6-a242e0e2a513)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

```
![image](https://github.com/user-attachments/assets/5769e215-d152-45c7-852f-fa5fee4ba322)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/d48526e1-a363-4e54-8552-bc30cefd92f3)
```

df2
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/0a8af10d-f65a-4ad9-b31d-51794118e08a)
```
from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv(r"C:\Users\Suriya\Downloads\data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
```

![image](https://github.com/user-attachments/assets/afae9ba3-fec5-49a3-8780-cc3dd01209bb)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/aaeb0f36-351e-4f86-873e-a97e89b41405)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv(r"C:\Users\Suriya\Downloads\Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/5ebbc3a6-a7cf-4fa2-bf1d-30b6d2903774)
```

df.skew()
```
![image](https://github.com/user-attachments/assets/613d5669-aa6f-4e67-9eb3-ddb5101aae96)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/fc253be2-bb26-45dc-9b32-b497651a0fc3)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/de96cbb2-725c-4d24-ac5c-bb0b41657075)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/3f7f54f8-ad39-4d80-912b-1fd9aeb46c02)
```

np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/6daa2831-f1bd-42d1-ac16-7ee9f076d9da)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/5c05d5f5-a07e-41e8-ab1a-fc69e2db5366)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/a03be3ea-f68b-470b-813a-530c5ea4a2dc)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/19c19f18-2129-41af-9c67-161e96c2f105)
```
df.skew()
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/9b075c04-4b0e-4970-a1f1-50b955e36b33)
```

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```
![image](https://github.com/user-attachments/assets/768055a6-16b1-4185-9136-735e4645c7dc)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/43a2a9c7-3211-4c40-98dd-f37067305ccf)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ba2c6c5d-8261-4b52-b5cc-ad913ec25b6c)
```
plt.show()
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/00434f39-9cd3-48a2-93d4-2d459dd87550)

 
# RESULT:
  program has been executed successfully.

       
