Setup & Run
----
- Create an environment  
`$ conda create -n <your env name> python=3.7 anaconda`  
- Activate your environment  
`$ source activate <your env name>`  
- Change Directory  
`(<your env name>) $ cd market-watcher`  
- Install dependent packages  
`(<your env name>) market-watcher $ pip install -r requirements.txt`  
- Run market-watcher  
`(<your env name>) market-watcher $ bokeh serve --show main.py`  
