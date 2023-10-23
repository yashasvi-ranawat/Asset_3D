from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from generate_glb import main as main_glb
from generate_latents import main as main_latents
import os


app = FastAPI()


@app.get("/text/{prompt}")
async def prediction(prompt: str):
    prompt = prompt.lower()
    if os.path.exists("prompt.txt"):
        raise HTTPException(
            status_code=503,
            detail="Server handling other request, retry after 3 minutes"
        )
    if len(prompt) > 150:
        raise HTTPException(
            status_code=414,
            detail="Prompt is too long"
        )
    filename = f"cache/{prompt}.glb"
    os.makedirs("cache", exist_ok=True)
    if not os.path.exists(filename):
        try:
            with open("prompt.txt", "w") as fio:
                fio.write(prompt)
            latents = main_latents(prompt)
            glb_bin = main_glb(latents)

            with open(filename, "wb") as fio:
                fio.write(glb_bin)
            os.remove("prompt.txt")
        except Exception as e:
            os.remove("prompt.txt")
            raise HTTPException(status_code=500, detail=e.__str__())

    return FileResponse(
        filename,
        headers={
            "content-disposition": f"attachment; filename = {os.path.basename(filename)}"
        }
    )
