import pandas as pd
import numpy as np

def pxy(x, mean, var):
	p = np.exp((-(x-mean)**2) / (2*var)) / np.sqrt(2*np.pi*var)
	return p

train = pd.DataFrame()
train['Gender'] = ['male','male','male','male','female','female','female','female']
train['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
train['Weight'] = [180,190,170,165,100,150,130,150]
train['Foot_Size'] = [12,11,12,10,6,8,7,9]

test = pd.DataFrame()
test['Height'] = [6]
test['Weight'] = [180]
test['Foot_Size'] = [12]

P_male = (train['Gender'][train['Gender'] == 'male'].count()) / (train['Gender'].count())
P_female = (train['Gender'][train['Gender'] == 'female'].count()) / (train['Gender'].count())

train_means = train.groupby('Gender').mean()
train_variance = train.groupby('Gender').var()

print(train_means,'\n', train_variance)

pred_male = P_male * \
			pxy(test['Height'][0], train_means['Height'][1], train_variance['Height'][1]) * \
			pxy(test['Weight'][0], train_means['Weight'][1], train_variance['Weight'][1]) * \
			pxy(test['Foot_Size'][0], train_means['Foot_Size'][1], train_variance['Weight'][1])
			
pred_female = P_female * \
			pxy(test['Height'][0], train_means['Height'][0], train_variance['Height'][0]) * \
			pxy(test['Weight'][0], train_means['Weight'][0], train_variance['Weight'][0]) * \
			pxy(test['Foot_Size'][0], train_means['Foot_Size'][0], train_variance['Weight'][0])
print('pred_male:',pred_male,'\npred_female:',pred_female)		
print('Prediction: ', ('Male' if pred_male > pred_female else 'Female'))
