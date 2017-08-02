import asyncio
import dscframework
import codecs, json
from keras.models import load_model
from bbox import extract
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Parameters
version = 1

# Model
model = load_model(("export/mdl_v%d.h5")%(version))

# Client
cli = dscframework.Client("ws://localhost:8080")

def encode_input(data):
    return []

def predict(data):
    batch = encode_input(data)
    result = model.predict(batch, batch_size=len(batch), verbose=0)
    output = extract(result)
    return result

async def on_message(head, data):
    output = predict(data)
    await cli.broadcast("vision", head, json.dumps(output))

async def on_update():
    pass

async def on_connect(cli):
    await cli.subscribe("camera", on_message)
    await cli.register("vision", {"output": {}, "input": {}}, on_update)
    print("Running Vision Sensor", flush=True)

async def main():
    await cli.start(on_connect)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
