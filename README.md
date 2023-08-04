<!---
Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.
-->

<p align="center">
  <img src="https://docs.periflow.ai/img/logo.svg" width="80%" alt="system">
</p>

<h2><p align="center">Supercharge Generative AI Serving 🚀</p></h2>

<p align="center">
  <a href="https://github.com/friendliai/periflow-client/actions/workflows/ci.yaml">
    <img alt="CI Status" src="https://github.com/friendliai/periflow-client/actions/workflows/ci.yaml/badge.svg">
  </a>
  <a href="https://pypi.org/project/periflow-client/">
    <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/periflow-client?logo=Python&logoColor=white">
  </a>
  <a href="https://pypi.org/project/periflow-client/">
      <img alt="PyPi Package Version" src="https://img.shields.io/pypi/v/periflow-client?logo=PyPI&logoColor=white">
  </a>
  <a href="https://docs.periflow.ai/">
    <img alt="Documentation" src="https://img.shields.io/badge/read-doc-blue?logo=ReadMe&logoColor=white">
  </a>
  <a href="https://github.com/friendliai/periflow-client/blob/main/LICENSE">
      <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=Apache">
  </a>
</p>

PeriFlow is the fastest engine for serving generative AI models such as GPT-3. With PeriFlow, a company can significantly reduce the cost and environmental impact of running its generative AI models. Users can use PeriFlow in a container and run it on the infrastructure they manage. They can also use our PeriFlow cloud service to reduce overheads of running generative AI models themselves.

As these models enable smarter, more productive services, many companies are competitively investing to build sophisticated generative AI models, with parameter sizes ranging from a few billion to even a few trillion. OpenAI, for example, is well-known for its large language models such as ChatGPT and GPT-4. There are many others as well—LLaMA, OPT, BLOOM, Gopher, PaLM, BlenderBot3, Codex, and CodeGen, to name a few. Many companies are increasingly making their generative AI models or adapting existing models to their needs.

Utilizing such large generative AI models, however, is no simple task. Even after users complete training models, they still face big challenges while handling the inference. Namely, the serving costs are likely high since they require GPUs—and as traffic increases, the costs steeply increase. Furthermore, they could pose environmental consequences. Lastly, operating the serving infrastructure can be burdensome for users who want to solely focus on training their models.

This is where PeriFlow, an engine for generative AI, comes to the rescue. Not only does PeriFlow provide a huge speedup of serving generative AI models, but it also provides a way for running the engine in diverse cloud or on-premise GPU resources. Namely, the PeriFlow cloud service automatically scales depending on the traffic and handles any faults or performance problems. Just imagine the time and cost that could be saved by leaving all tasks for utilizing a generative AI model in the hands of PeriFlow.

# ☁️ PeriFlow Cloud

## High performance

Users can use PeriFlow to reduce serving costs and environmental consequences significantly. They can serve much higher traffic with the same number of GPUs—or serve the same amount of traffic with notably fewer GPUs. PeriFlow can serve 10x more throughput at the same level of latency. This gain is thanks to PeriFlow’s innovative patented technology, which speeds up the execution of generative AI models.

## Diverse model and options support

PeriFlow supports various language model architectures, embedding choices, and decoding options such as greedy decoding, top-k, top-p, and beam search. PeriFlow will support diffusion models as well in the near future, so stay tuned!
Users can use PeriFlow in a container and run it by themselves, or they can use our cloud service. The cloud service supports the following features.

## Effortless deployment

PeriFlow cloud provides an easy serving experience with a Command Line Interface (CLI) and a web interface. With just a few clicks, users can deploy their models to the infrastructure that they desire. Users can move their serving between different clouds such as Azure, AWS, and GCP, and still have the same seamless experience.

## Automatic load and fault management

PeriFlow cloud monitors the resources in use and requests (responses) being sent to (sent from) the currently deployed model, allowing users a more stable model serving experience. When the number of requests sent to the deployed model increases, PeriFlow cloud automatically assigns more resources (GPU VMs) to the model, while it reduces resource usage when there are not as many requests. Furthermore, if a certain resource malfunctions, PeriFlow cloud proceeds with recovery based on the monitoring results.

# 🕹️ PeriFlow Client

Check out [PeriFlow Client Docs](https://docs.periflow.ai/) to learn more.

## Installation

```sh
pip install periflow-client
```

If you have a Hugging Face checkpoint and want to convert it to a PeriFlow-compatible format, you need to install the package with the necessary machine learing library (`mllib`) dependencies. In this case, install the package with the following command:

```sh
pip install periflow-client[mllib]
```

## Examples

This example shows how to create a deployment and send a completion API request to the created deployment with Python SDK.

```python
import periflow as pf


# Set up PeriFlow context.
pf.init(
    api_key="YOUR_PERIFLOW_API_KEY",
    project_name="my-project",
)

# Create a deployment at GCP asia-northest3 region wtih one A100 GPU.
deployment = pf.Deployment.create(
    checkpoint_id="YOUR_CHECKPOINT_ID",
    name="my-deployment",
    cloud="gcp",
    region="asia-northeast3",
    vm_type="a2-highgpu-1g",
    ...
)
```

When the deployment becomes the "Healthy" status and ready to process inference requests, you can generate a completion with:

```python
# Generate a completion by sending an inference request to the deployment created above.
api = pf.Completion(deployment_id=deployment.deployment_id)
completion = api.create(
    options=pf.V1CompletionOptions(
        prompt="Python is a popular language for",
        max_tokens=100,
        top_p=0.8,
        temperature=0.5,
        no_repeat_ngram=3,
    )
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
pf project switch my-project

# Create a deployment
pf deployment create \
  --checkpoint-id $YOUR_CHECKPOINT_ID \
  --name my-deployment \
  --cloud gcp \
  --region asia-northeast3 \
  --vm-type a2-highgpu-1g \
  --config-file config.yaml
```

When the deployment is ready, you can send a request with `curl`.

```sh
# Send a inference request to the deployment.
curl -X POST https://gcp-asia-northeast3.periflow.ai/$DEPLOYMENT_ID/v1/completions \
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
