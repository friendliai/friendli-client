<!---
Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.
-->

<p align="center">
  <img src="https://docs.friendli.ai/img/logo.svg" width="80%" alt="system">
</p>

<h2><p align="center">Supercharge Generative AI Serving 🚀</p></h2>

<p align="center">
  <a href="https://github.com/friendliai/friendli-client/actions/workflows/ci.yaml">
    <img alt="CI Status" src="https://github.com/friendliai/friendli-client/actions/workflows/ci.yaml/badge.svg">
  </a>
  <a href="https://pypi.org/project/friendli-client/">
    <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/friendli-client?logo=Python&logoColor=white">
  </a>
  <a href="https://pypi.org/project/friendli-client/">
      <img alt="PyPi Package Version" src="https://img.shields.io/pypi/v/friendli-client?logo=PyPI&logoColor=white">
  </a>
  <a href="https://docs.friendli.ai/">
    <img alt="Documentation" src="https://img.shields.io/badge/read-doc-blue?logo=ReadMe&logoColor=white">
  </a>
  <a href="https://github.com/friendliai/friendli-client/blob/main/LICENSE">
      <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=Apache">
  </a>
</p>

Friendli engine is the fastest engine for serving generative AI models such as GPT-3. With Friendli Suite, a company can significantly reduce the cost and environmental impact of running its generative AI models. Users can use Friendli engine in a container and run it on the infrastructure they manage. They can also use our Friendli dedicated endpoint service to reduce overheads of running generative AI models themselves.

# Friendli Suite

## High performance

Users can use Friendli to reduce serving costs and environmental consequences significantly. They can serve much higher traffic with the same number of GPUs—or serve the same amount of traffic with notably fewer GPUs. Friendli can serve 10x more throughput at the same level of latency.

## Diverse model and options support

Friendli supports various language model architectures, embedding choices, and decoding options such as greedy decoding, top-k, top-p, and beam search. Friendli will support diffusion models as well in the near future, so stay tuned!
Users can use Friendli in a container and run it by themselves, or they can use our cloud service. The cloud service supports the following features.

## Effortless deployment

Friendli dedicated endpoints provides an easy serving experience with a Command Line Interface (CLI) and a web interface. With just a few clicks, users can deploy their models to the infrastructure that they desire. Users can move their serving between different clouds such as Azure, AWS, and GCP, and still have the same seamless experience.

## Automatic load and fault management

Friendli dedicated endpoints monitor the resources in use and requests (responses) being sent to (sent from) the currently deployed model, allowing users a more stable model serving experience. When the number of requests sent to the deployed model increases, it automatically assigns more resources (GPU VMs) to the model, while it reduces resource usage when there are not as many requests. Furthermore, if a certain resource malfunctions, it proceeds with recovery based on the monitoring results.

# 🕹️ Friendli Client

Check out [Friendli Client Docs](https://docs.friendli.ai/) to learn more.

## Installation

```sh
pip install friendli-client
```

If you have a Hugging Face checkpoint and want to convert it to a Friendli-compatible format with applying quantization, you need to install the package with the necessary machine learing library (`mllib`) dependencies.
In this case, install the package with the following command:

```sh
pip install "friendli-client[mllib]"
```

## Examples

This example shows how to create a deployment and send a completion API request to the created deployment with Python SDK.

```python
import os
from friendli import FriendliResource

client = FriendliResource(
    api_key=os.environ["FRIENDLI_API_KEY"],
    project=os.environ["FRIENDLI_PROJECT"],
)

# Create a deployment at GCP asia-northest3 region wtih one A100 GPU.
deployment = client.deployment.create(
    checkpoint_id=os.environ["CHECKPOINT_ID"],
    name="my-deployment",
    cloud="gcp",
    region="asia-northeast3",
    gpu_type="a100",
    num_gpus=1,
)
```

When the deployment becomes the "Healthy" status and ready to process inference requests, you can generate a completion with:

```python
from friendli import Friendli

client = Friendli(
    api_key=os.environ["FRIENDLI_API_KEY"],
    project=os.environ["FRIENDLI_PROJECT"],
    deployment_id=os.environ["DEPLOYMENT_ID"],
)

# Generate a completion by sending an inference request to the deployment created above.
completion = client.completions.create(
    prompt="Python is a popular language for",
    max_tokens=100,
    top_p=0.8,
    temperature=0.5,
    no_repeat_ngram=3,
)
print(completion.choices[0].text)

"""
>>> Example Output:

web development. It is also used for a variety of other applications.
Python can be used to create desktop applications, web applications and mobile applications as well.
Python is one of the most popular languages for data science.
Data scientists use Python to analyze data.
The Python ecosystem is very diverse.
There are many libraries that can help you with your Python projects.
You can also find many Python tutorials online.
"""
```

You can also do the same with CLI.

```sh
# Switch CLI context to target project
friendli project switch my-project

# Create a deployment
friendli deployment create \
  --checkpoint-id $YOUR_CHECKPOINT_ID \
  --name my-deployment \
  --cloud gcp \
  --region asia-northeast3 \
  --gpu-type a100 \
  --num-gpus 1 \
  --config-file config.yaml
```

When the deployment is ready, you can send a request with `curl`.

```sh
# Send a inference request to the deployment.
curl -X POST https://gcp-asia-northeast3.friendli.ai/$DEPLOYMENT_ID/v1/completions \
  -d '{"prompt": "Python is a popular language for", "max_tokens": 100, "top_p": 0.8, "temperature": 0.5, "no_repeat_ngram": 3}'
```

The response will be like:

```txt
{
   "choices": [
      {
         "index": 0,
         "seed": 18337142367832222086,
         "text": " web development. It is also used for a variety of other applications.\nPython can be used to create desktop applications, web applications and mobile applications as well.\nPython is one of the most popular languages for data science.\nData scientists use Python to analyze data.\nThe Python ecosystem is very diverse.\nThere are many libraries that can help you with your Python projects.\nYou can also find many Python tutorials online.
         "tokens": [3644,8300,290,3992,2478,13,198,37906,318,6768,973,284,...]
      }
   ]
}
```
