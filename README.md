# chatbot-api

SEE [examples.ipynb](examples.ipynb) for detailed examples.

Now support:

* [llama](https://huggingface.co/decapoda-research/llama-7b-hf) with [lora](https://huggingface.co/tloen/alpaca-lora-7b). `cfgs/llama-7b-lora.json`
* [chatglm](https://huggingface.co/THUDM/chatglm-6b). `cfgs/chatglm-6b.json`
* [blip2chatglm](https://github.com/XiPotatonium/LAVIS). `cfgs/blip2zh-chatglm-6b.json`. Currently only training code is provided, we will release pretrained model soon.

# Run

```
uvicorn src:app --reload
```
