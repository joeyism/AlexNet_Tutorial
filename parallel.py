from tqdm import tqdm
import multiprocessing as mp
import time

cores = mp.cpu_count()

def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = mp.Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def tqdm_parallelize(data, func, desc="Running in parallel", unit="it"):
    with mp.Pool(processes=cores) as p:
        result = list(tqdm(p.imap(func, data), desc=desc, total=len(data), unit=unit))

    return result

