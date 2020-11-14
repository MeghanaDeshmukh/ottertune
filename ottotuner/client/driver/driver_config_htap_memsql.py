import os

#==========================================================
#  HOST LOGIN
#==========================================================

# Location of the database host relative to this driver
# Valid values: local, remote, docker or remote_docker
HOST_CONN = 'local'

# The name of the Docker container for the target database
# (only required if HOST_CONN = docker)
CONTAINER_NAME = 'postgres_container' # e.g., 'postgres_container'

# Host SSH login credentials (only required if HOST_CONN=remote)
LOGIN_NAME = None
LOGIN_HOST = None
LOGIN_PASSWORD = None
LOGIN_PORT = None  # Set when using a port other than the SSH default


#==========================================================
#  DATABASE OPTIONS
#==========================================================

# Postgres, Oracle or Mysql
DB_TYPE = 'memsql'

# Database version
DB_VERSION = '14'

# Name of the database
DB_NAME = 'htap'

# Database username
DB_USER = 'root'

# Password for DB_USER
DB_PASSWORD = 'ottotuner'

# Database admin username (for tasks like restarting the database)
ADMIN_USER = DB_USER

# Database host address
DB_HOST = '127.0.0.1'

# Database port
DB_PORT = '3318'

# If set to True, DB_CONF file is mounted to database container file
# Only available when HOST_CONN is docker or remote_docker
DB_CONF_MOUNT = False

# Path to the configuration file on the database server
# If DB_CONF_MOUNT is True, the path is on the host server, not docker
DB_CONF = '/etc/sysctl.conf'

# Path to the directory for storing database dump files
DB_DUMP_DIR = '/tmp/htap/'

# Base config settings to always include when installing new configurations
if DB_TYPE == 'mysql':
    BASE_DB_CONF = {
        'innodb_monitor_enable': 'all',
        # Do not generate binlog, otherwise the disk space will grow continuely during the tuning
        # Be careful about it when tuning a production database, it changes your binlog behavior.
        'skip-log-bin': None,
    }
elif DB_TYPE == 'memsql':
    BASE_DB_CONF = None
        # Do not generate binlog, otherwise the disk space will grow continuely during the tuning
        # Be careful about it when tuning a production database, it changes your binlog behavior.
elif DB_TYPE == 'postgres':
    BASE_DB_CONF = {
        'track_counts': 'on',
        'track_functions': 'all',
        'track_io_timing': 'on',
        'autovacuum': 'off',
    }
else:
    BASE_DB_CONF = None

# Name of the device on the database server to monitor the disk usage, or None to disable
DATABASE_DISK = None

# Set this to a different database version to override the current version
OVERRIDE_DB_VERSION = 14.14

# POSTGRES-SPECIFIC OPTIONS >>>
#PG_DATADIR = '/var/lib/postgresql/10/main'
PG_DATADIR = '/etc/lib/memsql/'

# ORACLE-SPECIFIC OPTIONS >>>
ORACLE_AWR_ENABLED = False
ORACLE_FLASH_BACK = True
RESTORE_POINT = 'htap_point'
RECOVERY_FILE_DEST = '/opt/oracle/oradata/ORCL'
RECOVERY_FILE_DEST_SIZE = '15G'


#==========================================================
#  DRIVER OPTIONS
#==========================================================

# Path to this driver
DRIVER_HOME = os.path.dirname(os.path.realpath(__file__))

# Path to the directory for storing results
RESULT_DIR = os.path.join(DRIVER_HOME, 'results')

# Set this to add user defined metrics
ENABLE_UDM = False

# Path to the User Defined Metrics (UDM), only required when ENABLE_UDM is True
UDM_DIR = os.path.join(DRIVER_HOME, 'userDefinedMetrics')

# Path to temp directory
TEMP_DIR = '/tmp/driver'

# Path to the directory for storing database dump files
if DB_DUMP_DIR is None:
    if HOST_CONN == 'local':
        DB_DUMP_DIR = os.path.join(DRIVER_HOME, 'dumpfiles')
        if not os.path.exists(DB_DUMP_DIR):
            os.mkdir(DB_DUMP_DIR)
    else:
        DB_DUMP_DIR = os.path.expanduser('~/')

# Reload the database after running this many iterations
RELOAD_INTERVAL = 10

# The maximum allowable disk usage percentage. Reload the database
# whenever the current disk usage exceeds this value.
MAX_DISK_USAGE = 90

# Execute this many warmup iterations before uploading the next result
# to the website
WARMUP_ITERATIONS = 0

# Let the database initialize for this many seconds after it restarts
RESTART_SLEEP_SEC = 30

#==========================================================
#  OLTPBENCHMARK OPTIONS
#==========================================================

# Path to OLTPBench directory
OLTPBENCH_HOME = os.path.expanduser('/home/mrd/Desktop/OttoTuner/htap/HTAPBench-master/')

# Path to the OLTPBench configuration file
OLTPBENCH_CONFIG = os.path.join(OLTPBENCH_HOME, 'config/htapb_config_memsql_ottotuner.xml')

# Name of the benchmark to run
OLTPBENCH_BENCH = 'htap'

# Name of the benchmark handler
RUN_CMD = 'java -cp /home/mrd/Desktop/OttoTuner/htap/HTAPBench-master/target/htapbench-0.95-jar-with-dependencies.jar pt.haslab.htapbench.core.HTAPBench --s 5' ## For htapBench
LOG_PATH = os.path.join(OLTPBENCH_HOME, 'htap_run.log') ## For htapBench
HTAP_FLAG = True
#RUN_CMD = '/home/mrd/Desktop/OttoTuner/otterTuneCode/oltpbench/oltpbenchmark'  ## For oltp and olap / tpcc and tpch

#==========================================================
#  CONTROLLER OPTIONS
#==========================================================

# Controller observation time, OLTPBench will be disabled for
# monitoring if the time is specified
CONTROLLER_OBSERVE_SEC = 100

# Path to the controller directory
CONTROLLER_HOME = DRIVER_HOME + '/../controller'
#CONTROLLER_HOME = '/home/mrd/Desktop/OttoTuner/otterTuneCode/ottertune/client/controller'

# Path to the controller configuration file
CONTROLLER_CONFIG = os.path.join(CONTROLLER_HOME, 'config/sample_memsql_config.json'.format(DB_TYPE))


#==========================================================
#  LOGGING OPTIONS
#==========================================================

LOG_LEVEL = 'DEBUG'

# Path to log directory
LOG_DIR = os.path.join(DRIVER_HOME, 'log')

# Log files
DRIVER_LOG = os.path.join(LOG_DIR, 'driver.log')
OLTPBENCH_LOG = os.path.join(LOG_DIR, 'oltpbench.log')
CONTROLLER_LOG = os.path.join(LOG_DIR, 'controller.log')


#==========================================================
#  WEBSITE OPTIONS
#==========================================================

# OtterTune website URL
WEBSITE_URL = 'http://127.0.0.1:8000'
#WEBSITE_URL = 'http://0.0.0.0:8000/'

# Code for uploading new results to the website
UPLOAD_CODE = 'OXH0VAQQXH1ORD8KMNI7'
