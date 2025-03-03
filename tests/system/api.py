from promptflow.core._serving.app import create_app
from fastapi.security import HTTPBearer
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

security = HTTPBearer()

flow_file_path = os.path.dirname(__file__)
app = create_app(engine="fastapi", flow_file_path=flow_file_path)

# uvicorn tests.system.api:app --workers 1 --port 5000