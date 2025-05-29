import numpy as np

def mi(X, Y):
    _, ar, _ = np.unique(X, return_index=True, return_inverse=True, axis=0)
    arity_X = len(ar)
    _, ar, _ = np.unique(Y, return_index=True,return_inverse=True)
    arity_Y = len(ar)

    n = len(Y)
    table_dim = [arity_X, arity_Y]
    
    
    p_XY = np.histogram2d(X, Y, bins=table_dim)[0] / n
    #p_XY = np.bincount(X, Y) / n
    p_X_p_Y = (np.outer(np.sum(p_XY, axis=1), np.sum(p_XY, axis=0))).ravel(order='F')
    p_XY = p_XY.ravel(order='F')

    id_non_zero = np.intersect1d(np.nonzero(p_XY), np.nonzero(p_X_p_Y))
    MI = np.sum(np.sum(p_XY[id_non_zero] * np.log(p_XY[id_non_zero] / p_X_p_Y[id_non_zero])))

    return MI



def MIM(X_data, Y_labels, topK):
    score_per_feature = [mi(X_data[:, i], Y_labels) for i in range(X_data.shape[1])]
    selectedFeatures = np.argsort(score_per_feature)[::-1][:topK]
    return selectedFeatures


def JMI(X_data, Y_labels, topK):
    numFeatures = X_data.shape[1]
    mi_score = np.array([mi(X_data[:, i], Y_labels) for i in range(numFeatures)])
    selectedFeatures = np.ones(topK, dtype=int)*-1
    selectedFeatures[0] = np.argmax(mi_score)
    notSelectedFeatures = np.setdiff1d(range(numFeatures), selectedFeatures)

    # Efficient implementation of the second step
    score_per_feature = np.zeros(numFeatures)
    score_per_feature[selectedFeatures[0]] = np.nan
    for count in range(1, topK):
        for index_feature_ns in notSelectedFeatures:
            _, _, X2 = np.unique(np.column_stack((X_data[:, index_feature_ns], X_data[:, selectedFeatures[count - 1]])), return_index=True, return_inverse=True, axis=0)
            score_per_feature[index_feature_ns] += mi(X2, Y_labels)

        selectedFeatures[count] = np.nanargmax(score_per_feature)
        score_per_feature[selectedFeatures[count]] = np.nan
        notSelectedFeatures = np.setdiff1d(range(numFeatures), selectedFeatures)

    return selectedFeatures