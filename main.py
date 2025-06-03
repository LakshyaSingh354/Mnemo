from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Agent Memory Store is running 🚀"}
