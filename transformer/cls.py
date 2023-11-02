import numpy as np
path='leaf_ro.txt'

catrgory=15
X=np.loadtxt(path)
result=np.zeros(catrgory)
ave=np.mean(X[:,1]==X[:,0])
print("overall",ave)
for idx in range(catrgory):
    list=(X[:,1]==idx)
    result[idx]=np.mean(X[:,0][list]==idx)
    # if(result[idx]<0.6):
    #     print(idx,'--',result[idx])

# result=np.hstack([np.mean(X[:,1]==X[:,0]),result])
print(result)
np.savetxt("cls_result.txt",result,fmt=['%s'],newline='\n')
