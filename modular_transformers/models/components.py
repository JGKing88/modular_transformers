import transformers

class LM(transformers.GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = transformers.GPT2LMHeadModel(config)

