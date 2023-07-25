import pandas as pd
import numpy as np




def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


def split_sequences_multistep(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(n_steps_in, len(sequences)-n_steps_out):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i-n_steps_in:i], sequences[i: i+ n_steps_out]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def get_data(lag = 27, reshape = True):
    df = pd.read_csv('data.csv')
    df.drop('Unnamed: 0', inplace = True, axis = 1)
    df.columns = ['Date', 'sfu']
    df.Date = pd.to_datetime(df.Date)
    train_data = df[(df['Date'].dt.year < 2003) & (df['Date'].dt.year > 1979)]
    val_data = df[(df['Date'].dt.year >= 2003) & (df['Date'].dt.year <= 2014)]  
    test_data = df[(df['Date'].dt.year >= 2014) & (df['Date'].dt.year <= 2020)]    
    train_X, train_y = split_sequence(np.array(train_data['sfu']), lag)
    val_X, val_y = split_sequence(np.array(val_data['sfu']), lag)
    test_X, test_y = split_sequence(np.array(test_data['sfu']), lag)
    if reshape:    
        train_X = train_X.reshape((train_X.shape[0], lag, 1))
        val_X = val_X.reshape((val_X.shape[0], lag, 1))
        test_X = test_X.reshape((test_X.shape[0], lag, 1))
    return train_X, train_y, val_X, val_y, test_X, test_y


def get_data_multistep(in_steps, out_steps, reshape = True):
    X, y = list(), list()
    df = pd.read_csv('data.csv')
    df.drop('Unnamed: 0', inplace = True, axis = 1)
    df.columns = ['Date', 'sfu']
    df.Date = pd.to_datetime(df.Date)
    train_data = df[(df['Date'].dt.year < 2003) & (df['Date'].dt.year > 1979)]
    validation_data = df[(df['Date'].dt.year >= 2003) & (df['Date'].dt.year <= 2014)] 
    test_data = df[(df['Date'].dt.year >= 2014) & (df['Date'].dt.year <= 2020)]    
    train_X, train_y = split_sequences_multistep(np.array(train_data['sfu']), in_steps, out_steps)
    val_X, val_y = split_sequences_multistep(np.array(validation_data['sfu']), in_steps, out_steps)
    test_X, test_y = split_sequences_multistep(np.array(test_data['sfu']), in_steps, out_steps)
    if reshape:    
        train_X = train_X.reshape((train_X.shape[0],in_steps,1))
        test_X = test_X.reshape((test_X.shape[0], in_steps, 1))
        val_X = val_X.reshape((val_X.shape[0], in_steps, 1))
    return train_X, train_y, val_X, val_y, test_X, test_y


def get_data_CNN(lag = 27, reshape = True):
    df = pd.read_csv('data.csv')
    df.drop('Unnamed: 0', inplace = True, axis = 1)
    df.columns = ['Date', 'sfu']
    df.Date = pd.to_datetime(df.Date)
    train_data = df[(df['Date'].dt.year < 2003) & (df['Date'].dt.year > 1979)]
    val_data = df[(df['Date'].dt.year >= 2003) & (df['Date'].dt.year <= 2014)]  
    test_data = df[(df['Date'].dt.year >= 2014) & (df['Date'].dt.year <= 2020)]    
    train_X, train_y = split_sequence(np.array(train_data['sfu']), lag)
    val_X, val_y = split_sequence(np.array(val_data['sfu']), lag)
    test_X, test_y = split_sequence(np.array(test_data['sfu']), lag)
    if reshape:    
        train_X = train_X.reshape((train_X.shape[0],1, lag, 1))
        val_X = val_X.reshape((val_X.shape[0],1, lag, 1))
        test_X = test_X.reshape((test_X.shape[0],1, lag, 1))
    return train_X, train_y, val_X, val_y, test_X, test_y


def get_CNN_data_multistep(in_steps, out_steps, reshape = True):
    X, y = list(), list()
    df = pd.read_csv('data.csv')
    df.drop('Unnamed: 0', inplace = True, axis = 1)
    df.columns = ['Date', 'sfu']
    df.Date = pd.to_datetime(df.Date)
    train_data = df[(df['Date'].dt.year < 2003) & (df['Date'].dt.year > 1979)]
    val_data = df[(df['Date'].dt.year >= 2003) & (df['Date'].dt.year <= 2014)]  
    test_data = df[(df['Date'].dt.year >= 2014) & (df['Date'].dt.year <= 2020)]     
    train_X, train_y = split_sequences_multistep(np.array(train_data['sfu']), in_steps, out_steps)
    val_X, val_y = split_sequences_multistep(np.array(val_data['sfu']), in_steps, out_steps)
    test_X, test_y = split_sequences_multistep(np.array(test_data['sfu']), in_steps, out_steps)
    if reshape:    
        train_X = train_X.reshape((train_X.shape[0],1, in_steps, 1))
        val_X = val_X.reshape((val_X.shape[0],1, in_steps, 1))
        test_X = test_X.reshape((test_X.shape[0],1, in_steps, 1))
    return train_X, train_y, val_X, val_y, test_X, test_y
    