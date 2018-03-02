from joblib import Parallel, delayed
import multiprocessing
    
# what are your inputs, and what operation do you want to 
# perform on each input. For example...
inputs = range(10) 
def processInput(i):
    return i * i

core_n = multiprocessing.cpu_count()
    
results = Parallel(n_jobs=core_n)(delayed(processInput)(i) for i in inputs)
print(results)
