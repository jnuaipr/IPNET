from tdc.multi_pred import GDA
from loguru import logger
data = GDA(name = 'disgenet',path="./doc/data")
split = data.get_split()
df = data.get_data()
logger.info(f"GDA: \n{df}")