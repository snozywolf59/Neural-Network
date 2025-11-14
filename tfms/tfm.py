import torch
import torch.nn as nn

class MyTransformer(nn.Module):
    def __init__(self, encoder, decoder, in_embeder, out_embeder):
        self.encoder = encoder
        self.decoder = decoder
        self.input_embeder = in_embeder
        self.output_embeder = out_embeder
    
    def forward(self, inp, outp):
        return self.decoder(self.encoder(self.input_embeder(inp)), self.output_embeder(outp))