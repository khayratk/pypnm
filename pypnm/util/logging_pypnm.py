import logging.handlers
import logging
from mpi4py import MPI

my_id = MPI.COMM_WORLD.Get_rank()

logger = logging.getLogger('pypnm')
logger.setLevel(logging.WARN)

formatter = logging.Formatter('%(filename)s:%(lineno)s - %(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler
#rfh = logging.handlers.RotatingFileHandler("log_pypnm" + "_proc_" + str(my_id) + ".bz2", maxBytes=1024*1024*20, backupCount=5, encoding='bz2')
#rfh.setLevel(logging.WARNING)
#rfh.setFormatter(formatter)
#logger.addHandler(rfh)

# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
ch.setFormatter(formatter)
logger.addHandler(ch)