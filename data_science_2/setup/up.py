import os
import time

os.system("docker-compose up -d")
time.sleep(3)
os.system("chmod -R 777 ./pgadmin/var/lib/pgadmin")
os.system("python3 automatic_table.py")
os.system("python3 customers_table.py")
os.system("python3 items_table.py")
os.system("python3 test.py")
os.system("python3 remove_duplicates.py")
os.system("python3 fusion.py")
