import os

os.system("docker-compose down -v")
os.system("docker system prune -af")
