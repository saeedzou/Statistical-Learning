import numpy as np
def train_val_test_split(path="notMNIST.npz", train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    with np.load(path) as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2    # 2 = C 
        negClass = 9    # 9 = J
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data.transpose(2,0,1)
        SubData = Data[dataIndx[:,0],:,:]/(Data.max())
        SubTarget = Target[dataIndx[:,0],:].reshape(-1,1)
        # Change 2 & 9 to 0 & 1
        SubTarget[SubTarget==posClass] = 1  # 1 = C 
        SubTarget[SubTarget==negClass] = 0  # 0 = J
        np.random.seed(521)
        randIndx = np.arange(SubData.shape[0])
        np.random.shuffle(randIndx)
        SubData, SubTarget = SubData[randIndx,:,:], SubTarget[randIndx]
        SubData = SubData.reshape(-1,28 * 28)
        SubTarget = SubTarget.astype(np.float32)
        SubData = SubData.astype(np.float32)
        data_len = SubTarget.shape[0]
        train_split = int(data_len*train_ratio)
        test_split = int(data_len*(1-test_ratio))
        trainData = SubData[:train_split,:]
        trainTarget = SubTarget[:train_split]
        validData = SubData[train_split:test_split,:]
        validTarget = SubTarget[train_split:test_split]
        testData  = SubData[test_split:,:]
        testTarget = SubTarget[test_split:]
    return (trainData, trainTarget), (validData, validTarget), (testData, testTarget)