from typing import Dict
import numpy as np

import ray

# Step 1: Create a Ray Dataset from in-memory Numpy arrays.
# You can also create a Ray Dataset from many other sources and file
# formats.
ds = ray.data.from_numpy(np.asarray(["Complete this", "for me"]))

# Step 2: Define a Predictor class for inference.
# Use a class to initialize the model just once in `__init__`
# and re-use it for inference across multiple batches.
class HuggingFacePredictor:
    def __init__(self):
        from transformers import pipeline
        # Initialize a pre-trained GPT2 Huggingface pipeline.
        self.model = pipeline("text-generation", model="gpt2")

    # Logic for inference on 1 batch of data.
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Get the predictions from the input batch.
        predictions = self.model(list(batch["data"]), max_length=20, num_return_sequences=1)
        # `predictions` is a list of length-one lists. For example:
        # [[{'generated_text': 'output_1'}], ..., [{'generated_text': 'output_2'}]]
        # Modify the output to get it into the following format instead:
        # ['output_1', 'output_2']
        batch["output"] = [sequences[0]["generated_text"] for sequences in predictions]
        return batch

# Use 2 parallel actors for inference. Each actor predicts on a
# different partition of data.
scale = ray.data.ActorPoolStrategy(size=2)
# Step 3: Map the Predictor over the Dataset to get predictions.
predictions = ds.map_batches(HuggingFacePredictor, compute=scale)
# Step 4: Show one prediction output.
predictions.show(limit=1)