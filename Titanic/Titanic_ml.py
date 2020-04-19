import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model


data_train =  pd.read_csv(r"C:\Users\Administrator\Desktop\py学习\ml\Titanic\data\train.csv")

# 使用RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值特征提取出来，丢尽random forest regressor中
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    # 把乘客分为已知年龄和未知年龄两部分
    known_age =  age_df[age_df.Age.notnull()].loc[:,:].values
    unknow_age =  age_df[age_df.Age.isnull()].loc[:,:].values

    # y就是目标年龄
    y = known_age[:,0]
    # x即是特征属性值
    x = known_age[:,1:]

    # fit到randomforesttregressor中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x,y)

    # 用得到的模型进行未知年龄结果的预测
    predictedAges =  rfr.predict(unknow_age[:,1:])
    # 用得到的预测模型的结果填补原缺失的数据
    df.loc[(df.Age.isnull()),'Age'] = predictedAges

    return df,rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    return df

data_train,rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

# 将类别特征进行因子化
dummies_Cabin =  pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked =  pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex =  pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_Pclass =  pd.get_dummies(data_train['Pclass'],prefix='Pclass')
df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis=1)
df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis =1,inplace=True)

# 对Age和Fare两个属性进行缩放
scaler = preprocessing.StandardScaler()
age_scale_param =  scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

# 使用正则取出我们想要的属性值
train_df =  df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
print(train_df.dtypes)

train_np =  train_df.loc[:,:].values
# y即是Survival的结果
y = train_np[:,0]
x = train_np[:,1:]

# fit到模型中
clf =  linear_model.LogisticRegression(C=1.0,tol=1e-6)
clf.fit(x,y)

# 对测试数据进行同样数据的处理
data_test =  pd.read_csv(r"C:\Users\Administrator\Desktop\py学习\ml\Titanic\data\test.csv")
data_test.loc[(data_test.Fare.isnull()),'Fare'] = '0'
# 接着我们对data_test做和train_test一样的操作
# 首先用同样的RandomForestRegressor模型填充丢失的年龄
tmp_df = data_test.loc[(data_test.Age.isnull()),['Age','Fare','Parch','SibSp','Pclass']]
null_age =  tmp_df.loc[:,:].values
# 取出年龄的样本数据
other_data = null_age[:,1:]
# 进行样训练
predictedAges =  rfr.predict(other_data)
data_test.loc[(data_test.Age.isnull()),'Age'] = predictedAges

# 对Cabin进行处理
data_test =  set_Cabin_type(data_test)

# 对类别特征进行one-hot编码
dummies_Cabin =  pd.get_dummies(data_test['Cabin'],prefix='Cabin')
dummies_Embarked =  pd.get_dummies(data_test['Embarked'],prefix='Embarked')
dummies_Sex =  pd.get_dummies(data_test['Sex'],prefix='Sex')
dummies_Pclass =  pd.get_dummies(data_test['Pclass'],prefix='Pclass')

data_test = pd.concat([data_test,dummies_Cabin,dummies_Pclass,dummies_Sex,dummies_Embarked],axis=1)
data_test.drop(['Pclass','Name','Ticket','Cabin','Embarked'],axis=1,inplace=True)

# 对age,fare进行数据缩放 df['Age'].values.reshape(-1, 1)
data_test['Age_scaled'] =  scaler.fit_transform(data_test['Age'].values.reshape(-1, 1),age_scale_param)
data_test['Fare_scaled']  = scaler.fit_transform(data_test['Fare'].values.reshape(-1, 1),fare_scale_param)

# 将数据导入模型中，进行预测结果
test_df =  data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predicitions =  clf.predict(test_df)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'], 'Survived':predicitions.astype(np.int32)})
print(result)

