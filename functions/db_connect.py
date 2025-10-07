import os
import pg8000
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database credentials from environment variables
db_username = os.getenv('DB_USER')
db_password = os.getenv('DB_PASS')
db_host = os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')
db_port = os.getenv('DB_PORT')

#rds database
bastion_host =      os.getenv('BASTION_IP')
bastion_user =      os.getenv('BASTION_USERNAME')
bastion_password =  os.getenv('BASTION_PASSWORD')
rds_host     =      os.getenv('RDS_HOST')
rds_db_name      =      os.getenv('RDS_DB_NAME')
rds_user      =      os.getenv('RDS_DB_USER')
rds_password  =      os.getenv('RDS_DB_PASSWORD')

#temp vps database
def db_connect(db_name):
    try:
        connection = pg8000.connect(
            user=db_username,
            password=db_password,
            host=db_host,
            port=db_port,
            database=db_name
        )
        # print("Connected to the database")
        return connection
    except Exception as error:
        print(f"Error connecting to the database: {error}")
        return None

#rds database
import pg8000
import ssl
from sshtunnel import SSHTunnelForwarder

class SSHDBConnection:
    """
    Wraps an SSHTunnelForwarder + pg8000.Connection.
    Delegates all methods/attrs to the inner conn,
    and closes both conn+tunnel on .close().
    """

    def __init__(self, db_name: str):
        # Start SSH tunnel
        self._tunnel = SSHTunnelForwarder(
            (bastion_host, 22),
            ssh_username=bastion_user,
            ssh_password=bastion_password,
            remote_bind_address=(rds_host, 5432),
            local_bind_address=("127.0.0.1", 0),  # auto-pick free port
        )
        self._tunnel.start()

        # SSL-prefer logic for pg8000
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        try:
            # Try SSL first
            self._conn = pg8000.connect(
                user=rds_user,
                password=rds_password,
                host="127.0.0.1",
                port=self._tunnel.local_bind_port,
                database=db_name,
                ssl_context=ssl_ctx
            )
        except Exception as e:
            # Fallback to non-SSL if SSL fails
            self._conn = pg8000.connect(
                user=rds_user,
                password=rds_password,
                host="127.0.0.1",
                port=self._tunnel.local_bind_port,
                database=db_name,
                ssl_context=None
            )

    def close(self):
        """Close the DB socket, then tear down the SSH tunnel."""
        try:
            self._conn.close()
        finally:
            self._tunnel.stop()

    def __getattr__(self, name):
        # Delegate everything else to the pg8000 connection
        return getattr(self._conn, name)

def rds_connect(db_name=rds_db_name) -> SSHDBConnection:
    return SSHDBConnection(db_name)