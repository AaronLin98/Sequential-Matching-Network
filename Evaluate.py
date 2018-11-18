def ComputeR10_1(scores,labels,count = 10):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if max(sublist) == scores[i]:
                correct = correct + 1
    computeR10_1 = float(correct)/ total 
    print(computeR10_1)
    return computeR10_1 

def ComputeR2_1(scores,labels,count = 2):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if max(sublist) == scores[i]:
                correct = correct + 1
    # print(float(correct)/ total )
    computeR2_1 = float(correct)/ total 
    print(computeR2_1)
    return computeR2_1