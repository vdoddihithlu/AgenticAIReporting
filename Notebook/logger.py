import logging
import os   
import sys
from datetime import datetime


Log_file=f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"Logs",f"{datetime.now().strftime('%Y_%m_%d')}")
os.makedirs(log_path,exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_path,Log_file),
    format= "[ %(asctime)s ] %(levelname)s - %(name)s  %(lineno)d - %(message)s",
     level=logging.INFO,

)