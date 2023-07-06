#keep on adding testing.py 1 to testing200.py with cuda compiled opencv if you have a power full gpu
import os, time, sys
import concurrent.futures
from concurrent import futures

def run_process(process):
    print(process)
    os.system('python {}'.format(process))

   
if __name__ == "__main__":        
    processes = ['testing.py', 'testing2.py','testing3.py','testing100.py']
   
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = executor.map(run_process, processes)  
