from collections import defaultdict
def separate(a):
    split_index=[]
    for i in range(len(a)-1):
        if a[i]!=a[i+1]:
            split_index.append(i)
    return split_index


